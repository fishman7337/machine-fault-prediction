from fault_prediction.models import predict_scores, train_model


def test_fast_training_pipeline_predicts_probabilities(balanced_sample):
    result = train_model(balanced_sample, profile="fast", test_size=0.25, n_jobs=1)

    assert 0.0 <= result.metrics["threshold"] <= 1.0
    assert result.metrics["f1"] >= 0.0
    assert result.artifact["metadata"]["dropped_after_feature_engineering"]

    X = balanced_sample.drop(columns=["Machine Status"]).head(5)
    scores = predict_scores(result.artifact["model"], X)
    assert len(scores) == 5
    assert all(0.0 <= score <= 1.0 for score in scores)
