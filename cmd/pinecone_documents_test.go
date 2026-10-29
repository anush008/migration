//go:build !no_pinecone

package cmd

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/qdrant/migration/pkg/commons"
)

func parseDescription(t *testing.T, data string) *pineconeIndexDescription {
	t.Helper()
	var description pineconeIndexDescription
	require.NoError(t, json.Unmarshal([]byte(data), &description))
	return &description
}

func TestPineconeDocumentPlan(t *testing.T) {
	r := &MigrateFromPineconeCmd{SparseVector: "sparse_vector"}

	plan := r.documentPlan(parseDescription(t, `{"schema": {"fields": {
		"chunk_text": {"type": "string", "full_text_search": {"language": "en"},
			"embed": {"model": "llama-text-embed-v2", "dimension": 1024, "metric": "cosine", "write_parameters": {"input_type": "passage"}}},
		"title": {"type": "string", "full_text_search": {"language": "en"}}
	}}}`))
	require.NotNil(t, plan)
	require.Equal(t, []string{"chunk_text", "title"}, plan.Text)
	require.Len(t, plan.Vectors, 1)
	require.Equal(t, "chunk_text", plan.Vectors[0].Name, "one dense vector takes the name of its field")
	require.Equal(t, uint64(1024), plan.Vectors[0].Size)
	require.True(t, plan.Embedded)

	plan = r.documentPlan(parseDescription(t, `{"schema": {"fields": {
		"a": {"type": "dense_vector", "dimension": 3, "metric": "dotproduct"},
		"b": {"type": "dense_vector", "dimension": 4, "metric": "cosine"},
		"s": {"type": "sparse_vector"}
	}}}`))
	require.NotNil(t, plan)
	require.Equal(t, []string{"a", "b", "s"}, []string{plan.Vectors[0].Name, plan.Vectors[1].Name, plan.Vectors[2].Name})
	require.False(t, plan.Embedded)

	plan = r.documentPlan(parseDescription(t, `{"schema": {"fields": {
		"chunk_text": {"type": "string",
			"embed": {"model": "llama-text-embed-v2", "dimension": 1024, "metric": "cosine"},
			"sparse_embed": {"model": "pinecone-sparse-english-v0"}}
	}}}`))
	require.NotNil(t, plan)
	require.Equal(t, []string{"chunk_text", "chunk_text_sparse"}, []string{plan.Vectors[0].Name, plan.Vectors[1].Name},
		"a sparse vector from a field that also gives a dense vector gets the suffix _sparse")

	// Only the reserved vector fields and metadata: the vectors API serves the index.
	require.Nil(t, r.documentPlan(parseDescription(t, `{"schema": {"fields": {
		"_values": {"type": "dense_vector", "dimension": 8, "metric": "cosine"},
		"genre": {"type": "string"}
	}}}`)))
	require.Nil(t, r.documentPlan(parseDescription(t, `{"dimension": 8}`)))
}

func TestPineconeDocumentsToPoints(t *testing.T) {
	r := &MigrateFromPineconeCmd{SparseVector: "sparse_vector", IdField: "__id__", Pinecone: commons.PineconeConfig{Embed: "none"}}
	plan := r.documentPlan(parseDescription(t, `{"schema": {"fields": {
		"chunk_text": {"type": "string", "embed": {"model": "m", "dimension": 2, "metric": "cosine"}},
		"emb": {"type": "dense_vector", "dimension": 2, "metric": "cosine"},
		"sp": {"type": "sparse_vector"}
	}}}`))
	require.NotNil(t, plan)

	var documents map[string]map[string]any
	require.NoError(t, json.Unmarshal([]byte(`{
		"doc-1": {"_id": "doc-1", "chunk_text": "hello", "year": 2024, "emb": [0.5, 0.25], "sp": {"indices": [3, 9], "values": [0.1, 0.2]}},
		"doc-2": {"_id": "doc-2", "chunk_text": "world", "emb": null}
	}`), &documents))

	points, err := r.documentsToPoints(context.Background(), plan, []string{"doc-1", "doc-2", "doc-gone"}, documents, map[string]bool{})
	require.NoError(t, err)
	require.Len(t, points, 2, "a document deleted between list and fetch is skipped")

	first := points[0]
	require.Equal(t, "doc-1", first.Payload["__id__"].GetStringValue())
	require.Equal(t, "hello", first.Payload["chunk_text"].GetStringValue())
	require.NotContains(t, first.Payload, "emb")
	require.NotContains(t, first.Payload, "_id")
	vectors := first.Vectors.GetVectors().GetVectors()
	require.Equal(t, []float32{0.5, 0.25}, vectors["emb"].GetDense().GetData())
	require.Equal(t, []uint32{3, 9}, vectors["sp"].GetSparse().GetIndices())
	require.NotContains(t, vectors, "chunk_text", "with --pinecone.embed none, embedded fields get no vector")

	require.NotNil(t, points[1].Vectors, "a point without vectors still sends an empty map")
	require.Empty(t, points[1].Vectors.GetVectors().GetVectors())
}

func TestPineconeBaseURL(t *testing.T) {
	require.Equal(t, "https://example-index.svc.pinecone.io", pineconeBaseURL("example-index.svc.pinecone.io"))
	require.Equal(t, "http://localhost:5080", pineconeBaseURL("http://localhost:5080/"))
}
