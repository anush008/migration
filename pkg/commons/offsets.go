package commons

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/qdrant/go-client/qdrant"
)

func PrepareOffsetsCollection(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client) error {
	migrationOffsetCollectionExists, err := targetClient.CollectionExists(ctx, migrationOffsetsCollectionName)
	if err != nil {
		return fmt.Errorf("failed to check if collection exists: %w", err)
	}
	if migrationOffsetCollectionExists {
		return nil
	}
	return targetClient.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: migrationOffsetsCollectionName,
		VectorsConfig:  qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{}),
	})
}

func DeleteOffsetsCollection(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client) error {
	migrationOffsetCollectionExists, err := targetClient.CollectionExists(ctx, migrationOffsetsCollectionName)
	if err != nil {
		return fmt.Errorf("failed to check if collection exists: %w", err)
	}
	if !migrationOffsetCollectionExists {
		fmt.Printf("Collection %s does not exist, nothing to delete\n", migrationOffsetsCollectionName)
		return nil
	}
	return targetClient.DeleteCollection(ctx, migrationOffsetsCollectionName)
}

func GetStartOffset(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, sourceCollection string) (*qdrant.PointId, uint64, error) {
	point, err := getOffsetPoint(ctx, migrationOffsetsCollectionName, targetClient, sourceCollection)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to get start offset point: %w", err)
	}
	if point == nil {
		return nil, 0, nil
	}
	offset, ok := point.Payload[sourceCollection+"_offset"]
	if !ok {
		return nil, 0, nil
	}
	offsetCount, ok := point.Payload[sourceCollection+"_offsetCount"]
	if !ok {
		return nil, 0, nil
	}

	offsetCountValue, ok := offsetCount.GetKind().(*qdrant.Value_IntegerValue)
	if !ok {
		return nil, 0, fmt.Errorf("failed to get offset count: invalid type")
	}

	offsetID := getOffsetIdFromValue(offset)
	if offsetID == nil {
		return nil, 0, nil
	}
	return offsetID, uint64(offsetCountValue.IntegerValue), nil
}

// getOffsetIdFromValue is the inverse of getOffsetIdAsValue. It returns nil for unsupported values.
func getOffsetIdFromValue(value *qdrant.Value) *qdrant.PointId {
	switch v := value.GetKind().(type) {
	case *qdrant.Value_IntegerValue:
		return qdrant.NewIDNum(uint64(v.IntegerValue))
	case *qdrant.Value_StringValue:
		return qdrant.NewIDUUID(v.StringValue)
	default:
		return nil
	}
}

func getOffsetIdAsValue(offset *qdrant.PointId) (interface{}, error) {
	switch pointID := offset.GetPointIdOptions().(type) {
	case *qdrant.PointId_Num:
		return pointID.Num, nil
	case *qdrant.PointId_Uuid:
		return pointID.Uuid, nil
	default:
		return nil, fmt.Errorf("unsupported offset type: %T", pointID)
	}
}

func StoreStartOffset(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, sourceCollection string, offset *qdrant.PointId, offsetCount uint64) error {
	if offset == nil {
		return nil
	}
	offsetId, err := getOffsetIdAsValue(offset)
	if err != nil {
		return err
	}

	payload := qdrant.NewValueMap(map[string]any{
		sourceCollection + "_offset":       offsetId,
		sourceCollection + "_offsetCount":  offsetCount,
		sourceCollection + "_lastUpsertAt": time.Now().Format(time.RFC3339),
	})

	_, err = targetClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: migrationOffsetsCollectionName,
		Points: []*qdrant.PointStruct{
			{
				Id:      getOffsetPointId(sourceCollection),
				Payload: payload,
				Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
			},
		},
	})

	if err != nil {
		return fmt.Errorf("failed to store offset: %w", err)
	}
	return nil
}

func getOffsetPoint(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, sourceCollection string) (*qdrant.RetrievedPoint, error) {
	points, err := targetClient.Get(ctx, &qdrant.GetPoints{
		CollectionName: migrationOffsetsCollectionName,
		Ids:            []*qdrant.PointId{getOffsetPointId(sourceCollection)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get start offset: %w", err)
	}
	if len(points) == 0 {
		return nil, nil
	}

	return points[0], nil
}

func getOffsetPointId(sourceCollection string) *qdrant.PointId {
	deterministicUUID := uuid.NewSHA1(uuid.NameSpaceURL, []byte(sourceCollection))

	return qdrant.NewIDUUID(deterministicUUID.String())
}

// DeleteStartOffsets removes the offsets stored under the given keys.
func DeleteStartOffsets(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, keys []string) error {
	ids := make([]*qdrant.PointId, len(keys))
	for i, key := range keys {
		ids[i] = getOffsetPointId(key)
	}
	_, err := targetClient.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: migrationOffsetsCollectionName,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelector(ids...),
	})
	if err != nil {
		return fmt.Errorf("failed to delete offsets: %w", err)
	}
	return nil
}

// StoreBoundaries persists the range boundaries of a parallel migration under the given key.
func StoreBoundaries(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, key string, ids []*qdrant.PointId) error {
	values := make([]any, len(ids))
	for i, id := range ids {
		value, err := getOffsetIdAsValue(id)
		if err != nil {
			return err
		}
		values[i] = value
	}
	_, err := targetClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: migrationOffsetsCollectionName,
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{
			{
				Id:      getOffsetPointId(key),
				Payload: qdrant.NewValueMap(map[string]any{key + "_boundaries": values}),
				Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
			},
		},
	})
	if err != nil {
		return fmt.Errorf("failed to store range boundaries: %w", err)
	}
	return nil
}

// GetBoundaries loads the range boundaries stored by StoreBoundaries. It returns nil if none were stored.
func GetBoundaries(ctx context.Context, migrationOffsetsCollectionName string, targetClient *qdrant.Client, key string) ([]*qdrant.PointId, error) {
	point, err := getOffsetPoint(ctx, migrationOffsetsCollectionName, targetClient, key)
	if err != nil {
		return nil, fmt.Errorf("failed to get range boundaries: %w", err)
	}
	if point == nil {
		return nil, nil
	}
	values := point.Payload[key+"_boundaries"].GetListValue().GetValues()
	ids := make([]*qdrant.PointId, len(values))
	for i, value := range values {
		if ids[i] = getOffsetIdFromValue(value); ids[i] == nil {
			return nil, fmt.Errorf("invalid stored range boundary: %v", value)
		}
	}
	return ids, nil
}
