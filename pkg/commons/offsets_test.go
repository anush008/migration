package commons

import (
	"testing"

	"github.com/qdrant/go-client/qdrant"
)

func TestOffsetIdValueRoundTrip(t *testing.T) {
	for _, id := range []*qdrant.PointId{
		qdrant.NewIDNum(0),
		qdrant.NewIDNum(18446744073709551615),
		qdrant.NewIDUUID("0f4a6de3-c18b-5de3-992b-5eb7f5c52b1a"),
	} {
		v, err := getOffsetIdAsValue(id)
		if err != nil {
			t.Fatalf("to value %v: %v", id, err)
		}
		value, err := qdrant.NewValue(v)
		if err != nil {
			t.Fatalf("new value %v: %v", v, err)
		}
		got := getOffsetIdFromValue(value)
		if got.GetNum() != id.GetNum() || got.GetUuid() != id.GetUuid() {
			t.Fatalf("round trip mismatch: %v != %v", got, id)
		}
	}
	if getOffsetIdFromValue(qdrant.NewValueBool(true)) != nil {
		t.Fatal("expected nil for unsupported value")
	}
}
