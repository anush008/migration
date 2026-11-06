package cmd

import (
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_getPort(t *testing.T) {
	tests := []struct {
		name     string
		url      *url.URL
		expected int
	}{
		{
			name:     "tls enabled, custom port",
			url:      &url.URL{Scheme: "https", Host: "localhost:6334"},
			expected: 6334,
		},
		{
			name:     "tls enabled, default port",
			url:      &url.URL{Scheme: "https", Host: "localhost"},
			expected: 443,
		},
		{
			name:     "tls disabled, default port",
			url:      &url.URL{Scheme: "http", Host: "localhost"},
			expected: 80,
		},
		{
			name:     "tls disabled, custom port",
			url:      &url.URL{Scheme: "http", Host: "localhost:6334"},
			expected: 6334,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := getPort(tt.url)
			if got != tt.expected {
				t.Errorf("getPort() got = %v, expected %v", got, tt.expected)
			}
		})
	}
}

func TestQdrantSourceFilter(t *testing.T) {
	parse := func(args ...string) (*MigrateFromQdrantCmd, error) {
		ctx, err := NewParser(append([]string{"qdrant", "--source.collection", "source", "--target.collection", "target"}, args...))
		if err != nil {
			return nil, err
		}
		return ctx.Selected().Target.Addr().Interface().(*MigrateFromQdrantCmd), nil
	}

	unfiltered, err := parse()
	require.NoError(t, err)
	require.Nil(t, unfiltered.sourceFilter)
	// Without a filter, the offset keys stay as before, so an earlier migration can resume.
	require.Equal(t, "source", unfiltered.offsetsKey("source"))

	tenantA, err := parse("--source.filter", `{"must": [{"key": "tenant", "match": {"value": "a"}}]}`)
	require.NoError(t, err)
	require.NotNil(t, tenantA.sourceFilter)
	tenantB, err := parse("--source.filter", `{"must": [{"key": "tenant", "match": {"value": "b"}}]}`)
	require.NoError(t, err)
	require.NotEqual(t, "source", tenantA.offsetsKey("source"))
	require.NotEqual(t, tenantA.offsetsKey("source"), tenantB.offsetsKey("source"))

	_, err = parse("--source.filter", `{"must": [{"key": "tenant"}]}`)
	require.ErrorContains(t, err, "invalid filter")

	// A filter file resumes the same migration as the same filter on the command line.
	dir := t.TempDir()
	filterFile := filepath.Join(dir, "filter.json")
	require.NoError(t, os.WriteFile(filterFile, []byte(`{"must": [{"key": "tenant", "match": {"value": "a"}}]}`+"\n"), 0o600))
	fromFile, err := parse("--source.filter-file", filterFile)
	require.NoError(t, err)
	require.NotNil(t, fromFile.sourceFilter)
	require.Equal(t, tenantA.offsetsKey("source"), fromFile.offsetsKey("source"))

	emptyFile := filepath.Join(dir, "empty.json")
	require.NoError(t, os.WriteFile(emptyFile, nil, 0o600))
	_, err = parse("--source.filter-file", emptyFile)
	require.ErrorContains(t, err, "invalid filter")
	_, err = parse("--source.filter-file", filepath.Join(dir, "missing.json"))
	require.ErrorContains(t, err, "failed to read the filter file")
	// An empty path, for example from an unset shell variable, must not migrate all points.
	_, err = parse("--source.filter-file", "")
	require.ErrorContains(t, err, "failed to read the filter file")
	_, err = parse("--source.filter", "{}", "--source.filter-file", filterFile)
	require.ErrorContains(t, err, "can't be used together")
}
