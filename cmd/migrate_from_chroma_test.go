package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/qdrant/go-client/qdrant"

	"github.com/qdrant/migration/pkg/commons"
)

func TestConvertChromaValue(t *testing.T) {
	values := []any{json.Number("3"), json.Number("9007199254740993"), json.Number("2.0"), json.Number("1e3"), []any{json.Number("1"), "a"}}
	for i, value := range values {
		values[i] = convertChromaValue(value)
	}
	require.Equal(t, []any{int64(3), int64(9007199254740993), 2.0, 1000.0, []any{int64(1), "a"}}, values)
}

func TestChromaCollectionDistance(t *testing.T) {
	tests := []struct {
		name       string
		collection string
		expected   qdrant.Distance
	}{
		{"legacy metadata", `{"metadata": {"hnsw:space": "cosine"}, "configuration_json": {"hnsw": {"space": "l2"}}}`, qdrant.Distance_Cosine},
		{"hnsw", `{"configuration_json": {"hnsw": {"space": "ip"}}}`, qdrant.Distance_Dot},
		{"spann", `{"configuration_json": {"spann": {"space": "cosine"}}}`, qdrant.Distance_Cosine},
		{"default", `{"configuration_json": {}}`, qdrant.Distance_Euclid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var collection chromaCollection
			require.NoError(t, json.Unmarshal([]byte(tt.collection), &collection))
			distance, err := collection.distance()
			require.NoError(t, err)
			require.Equal(t, tt.expected, distance)
		})
	}
}

func TestChromaRequestRetriesWithAuth(t *testing.T) {
	var statuses []int
	var headers []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		headers = append(headers, req.Header.Get("Authorization"))
		status := []int{http.StatusServiceUnavailable, http.StatusOK}[len(statuses)]
		statuses = append(statuses, status)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`7`))
	}))
	defer server.Close()

	cmd := &MigrateFromChromaCmd{
		Chroma:     commons.ChromaConfig{Url: server.URL, AuthType: "token", Token: "secret", TokenHeader: "Authorization"},
		httpClient: server.Client(),
	}

	var count uint64
	require.NoError(t, cmd.request(context.Background(), http.MethodGet, "/count", nil, &count))
	require.Equal(t, uint64(7), count)
	require.Equal(t, []int{http.StatusServiceUnavailable, http.StatusOK}, statuses)
	require.Equal(t, []string{"Bearer secret", "Bearer secret"}, headers)
}
