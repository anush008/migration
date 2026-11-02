package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/pinecone-io/go-pinecone/v3/pinecone"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/qdrant/go-client/qdrant"
)

const (
	pineconeControlPlane = "https://api.pinecone.io"
	pineconeAPIVersion   = "2026-07"
)

// TestMigrateFromPineconeDocuments creates a Pinecone index with a document schema, so that the migration
// reads it through the documents API.
func TestMigrateFromPineconeDocuments(t *testing.T) {
	ctx := context.Background()
	apiKey := os.Getenv("PINECONE_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping Pinecone documents test. PINECONE_API_KEY not set")
	}

	indexName := fmt.Sprintf("migration-test-%d", rand.Int())
	pineconeAPI(ctx, t, apiKey, http.MethodPost, pineconeControlPlane+"/indexes", map[string]any{
		"name": indexName,
		"schema": map[string]any{"fields": map[string]any{
			"title":     map[string]any{"type": "string", "full_text_search": map[string]any{}},
			"body":      map[string]any{"type": "string", "full_text_search": map[string]any{}},
			"embedding": map[string]any{"type": "dense_vector", "dimension": dimension, "metric": "cosine"},
			"sparse":    map[string]any{"type": "sparse_vector"},
		}},
	}, nil)
	t.Cleanup(func() {
		pineconeAPI(context.WithoutCancel(ctx), t, apiKey, http.MethodDelete, pineconeControlPlane+"/indexes/"+indexName, nil, nil)
	})

	var index struct {
		Host   string `json:"host"`
		Status struct {
			Ready bool `json:"ready"`
		} `json:"status"`
	}
	require.Eventually(t, func() bool {
		pineconeAPI(ctx, t, apiKey, http.MethodGet, pineconeControlPlane+"/indexes/"+indexName, nil, &index)
		return index.Status.Ready
	}, 5*time.Minute, 2*time.Second, "index %q is not ready", indexName)

	// Only documents with an even number have a sparse vector.
	documents := make([]map[string]any, totalEntries)
	for i := range documents {
		documents[i] = map[string]any{
			"_id":       fmt.Sprintf("doc-%03d", i),
			"title":     fmt.Sprintf("title %d", i),
			"body":      fmt.Sprintf("body of document %d", i),
			"embedding": randFloat32Values(dimension),
			"year":      2000 + i,
			"tags":      []string{fmt.Sprintf("tag-%d", i%3)},
		}
		if i%2 == 0 {
			documents[i]["sparse"] = map[string]any{
				"indices": []uint32{uint32(i), uint32(i) + 100, uint32(i) + 200},
				"values":  randFloat32Values(3),
			}
		}
	}
	documentsURL := "https://" + index.Host + "/namespaces/__default__/documents/"
	pineconeAPI(ctx, t, apiKey, http.MethodPost, documentsURL+"upsert", map[string]any{"documents": documents}, nil)

	require.Eventually(t, func() bool {
		var stats struct {
			TotalVectorCount int `json:"totalVectorCount"`
		}
		pineconeAPI(ctx, t, apiKey, http.MethodPost, "https://"+index.Host+"/describe_index_stats", map[string]any{}, &stats)
		return stats.TotalVectorCount == totalEntries
	}, 2*time.Minute, 2*time.Second, "the documents of index %q are not readable", indexName)

	qdrantClient, qdrantHost, qdrantPort := startQdrant(ctx, t)

	collectionName := testCollectionName + "-" + indexName
	args := []string{
		"pinecone",
		fmt.Sprintf("--pinecone.index-name=%s", indexName),
		// The host has no scheme, as the Pinecone API returns it.
		fmt.Sprintf("--pinecone.index-host=%s", index.Host),
		fmt.Sprintf("--pinecone.api-key=%s", apiKey),
		fmt.Sprintf("--qdrant.url=http://%s:%d", qdrantHost, qdrantPort),
		fmt.Sprintf("--qdrant.collection=%s", collectionName),
		fmt.Sprintf("--qdrant.api-key=%s", qdrantAPIKey),
		fmt.Sprintf("--qdrant.id-field=%s", idField),
		"--migration.batch-size=7",
	}
	runMigrationBinary(t, args)

	info, err := qdrantClient.GetCollectionInfo(ctx, collectionName)
	require.NoError(t, err)
	dense := info.GetConfig().GetParams().GetVectorsConfig().GetParamsMap().GetMap()
	require.Len(t, dense, 1)
	require.Equal(t, uint64(dimension), dense["embedding"].GetSize())
	require.Equal(t, qdrant.Distance_Cosine, dense["embedding"].GetDistance())
	require.Contains(t, info.GetConfig().GetParams().GetSparseVectorsConfig().GetMap(), "sparse")
	for _, field := range []string{"title", "body"} {
		require.Equal(t, qdrant.PayloadSchemaType_Text, info.GetPayloadSchema()[field].GetDataType(), "text index on %q", field)
	}

	points, err := qdrantClient.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: collectionName,
		Limit:          qdrant.PtrOf(uint32(totalEntries + 1)),
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	require.NoError(t, err)
	require.Len(t, points, totalEntries)

	byID := make(map[string]*qdrant.RetrievedPoint, len(points))
	for _, point := range points {
		byID[point.Payload[idField].GetStringValue()] = point
	}
	for i, document := range documents {
		point, ok := byID[document["_id"].(string)]
		require.True(t, ok, "point for document %q not found", document["_id"])

		require.Equal(t, document["title"], point.Payload["title"].GetStringValue())
		require.Equal(t, document["body"], point.Payload["body"].GetStringValue())
		require.InDelta(t, float64(2000+i), point.Payload["year"].GetDoubleValue(), 0)
		require.Equal(t, document["tags"].([]string)[0], point.Payload["tags"].GetListValue().GetValues()[0].GetStringValue())
		require.NotContains(t, point.Payload, "embedding")
		require.NotContains(t, point.Payload, "sparse")

		vectors := point.Vectors.GetVectors().GetVectors()
		expected := normalizedVector(document["embedding"].([]float32))
		require.InDeltaSlice(t, expected, vectors["embedding"].GetDenseVector().GetData(), 1e-5)

		sparse, ok := document["sparse"].(map[string]any)
		if !ok {
			require.NotContains(t, vectors, "sparse", "document %q has no sparse vector", document["_id"])
			continue
		}
		actual := vectors["sparse"].GetSparseVector()
		require.ElementsMatch(t, sparse["indices"], actual.GetIndices())
		values := make(map[uint32]float32, len(actual.GetIndices()))
		for j, index := range actual.GetIndices() {
			values[index] = actual.GetValues()[j]
		}
		for j, index := range sparse["indices"].([]uint32) {
			require.InDelta(t, sparse["values"].([]float32)[j], values[index], 1e-6)
		}
	}
}

// TestMigrateFromPineconeVectors writes vectors to a new namespace of an existing Pinecone index, which the
// migration reads through the vectors API.
func TestMigrateFromPineconeVectors(t *testing.T) {
	ctx := context.Background()
	apiKey := os.Getenv("PINECONE_API_KEY")
	indexName := os.Getenv("PINECONE_INDEX_NAME")
	if apiKey == "" || indexName == "" {
		t.Skip("Skipping Pinecone vectors test. PINECONE_API_KEY and PINECONE_INDEX_NAME not set")
	}

	client, err := pinecone.NewClient(pinecone.NewClientParams{ApiKey: apiKey})
	require.NoError(t, err)
	index, err := client.DescribeIndex(ctx, indexName)
	require.NoError(t, err)
	require.Equal(t, int32(dimension), *index.Dimension, "index %q must have %d dimensions", indexName, dimension)

	namespace := fmt.Sprintf("migration-test-%d", rand.Int())
	indexConn, err := client.Index(pinecone.NewIndexConnParams{Host: index.Host, Namespace: namespace})
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, indexConn.DeleteAllVectorsInNamespace(context.WithoutCancel(ctx)))
	})

	ids := make([]string, totalEntries)
	vectors := make([]*pinecone.Vector, totalEntries)
	for i := range vectors {
		ids[i] = fmt.Sprintf("vec-%03d", i)
		metadata, err := structpb.NewStruct(map[string]any{"source": fmt.Sprintf("source-%d", i)})
		require.NoError(t, err)
		values := randFloat32Values(dimension)
		vectors[i] = &pinecone.Vector{Id: ids[i], Values: &values, Metadata: metadata}
	}
	_, err = indexConn.UpsertVectors(ctx, vectors)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		stats, err := indexConn.DescribeIndexStats(ctx)
		require.NoError(t, err)
		summary, ok := stats.Namespaces[namespace]
		return ok && summary.VectorCount == totalEntries
	}, 2*time.Minute, 2*time.Second, "the vectors of namespace %q are not readable", namespace)

	qdrantClient, qdrantHost, qdrantPort := startQdrant(ctx, t)

	collectionName := testCollectionName + "-" + namespace
	args := []string{
		"pinecone",
		fmt.Sprintf("--pinecone.index-name=%s", indexName),
		fmt.Sprintf("--pinecone.index-host=https://%s", index.Host),
		fmt.Sprintf("--pinecone.api-key=%s", apiKey),
		fmt.Sprintf("--pinecone.namespace=%s", namespace),
		fmt.Sprintf("--qdrant.url=http://%s:%d", qdrantHost, qdrantPort),
		fmt.Sprintf("--qdrant.collection=%s", collectionName),
		fmt.Sprintf("--qdrant.api-key=%s", qdrantAPIKey),
		fmt.Sprintf("--qdrant.id-field=%s", idField),
		"--migration.batch-size=7",
	}
	runMigrationBinary(t, args)

	points, err := qdrantClient.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: collectionName,
		Limit:          qdrant.PtrOf(uint32(totalEntries + 1)),
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	require.NoError(t, err)
	require.Len(t, points, totalEntries)

	byID := make(map[string]*qdrant.RetrievedPoint, len(points))
	for _, point := range points {
		byID[point.Payload[idField].GetStringValue()] = point
	}
	for i, id := range ids {
		point, ok := byID[id]
		require.True(t, ok, "point for vector %q not found", id)
		require.Equal(t, fmt.Sprintf("source-%d", i), point.Payload["source"].GetStringValue())
		expected := *vectors[i].Values
		if index.Metric == pinecone.Cosine {
			expected = normalizedVector(expected)
		}
		require.InDeltaSlice(t, expected, point.Vectors.GetVector().GetDenseVector().GetData(), 1e-5)
	}
}

func startQdrant(ctx context.Context, t *testing.T) (*qdrant.Client, string, int) {
	t.Helper()
	container := qdrantContainer(ctx, t, qdrantAPIKey)
	t.Cleanup(func() {
		require.NoError(t, container.Terminate(ctx))
	})
	host, err := container.Host(ctx)
	require.NoError(t, err)
	port, err := container.MappedPort(ctx, qdrantGRPCPort)
	require.NoError(t, err)
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   host,
		Port:                   int(port.Num()),
		APIKey:                 qdrantAPIKey,
		SkipCompatibilityCheck: true,
	})
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, client.Close())
	})
	return client, host, int(port.Num())
}

// pineconeAPI sends a request to the Pinecone REST API, which the Go client of this module does not
// support for indexes with a document schema.
func pineconeAPI(ctx context.Context, t *testing.T, apiKey, method, url string, body, out any) {
	t.Helper()
	var reader io.Reader
	if body != nil {
		payload, err := json.Marshal(body)
		require.NoError(t, err)
		reader = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	require.NoError(t, err)
	req.Header.Set("Api-Key", apiKey)
	req.Header.Set("X-Pinecone-API-Version", pineconeAPIVersion)
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	require.NoError(t, err)
	require.Less(t, res.StatusCode, 300, "%s %s returned status %d: %s", method, url, res.StatusCode, data)
	if out != nil {
		require.NoError(t, json.Unmarshal(data, out))
	}
}
