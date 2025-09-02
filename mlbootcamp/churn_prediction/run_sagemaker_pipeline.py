import argparse
import os
import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import MetricsSource
from sagemaker.model_metrics import MetricsSource as ModelMetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel


def get_session(region: str | None = None) -> Session:
    boto_sess = boto3.Session(region_name=region)
    sagemaker_sess = sagemaker.session.Session(boto_session=boto_sess)
    return sagemaker_sess


def build_pipeline(region: str, role_arn: str, bucket: str, base_job_prefix: str = "churn") -> Pipeline:
    sess = get_session(region)

    # Parameters
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    transform_instance_type = ParameterString(name="TransformInstanceType", default_value="ml.m5.large")
    train_samples = ParameterInteger(name="TrainSamples", default_value=7043)
    max_depth = ParameterInteger(name="MaxDepth", default_value=5)
    eta = ParameterFloat(name="Eta", default_value=0.2)
    num_round = ParameterInteger(name="NumRound", default_value=200)

    # Processing step (feature engineering)
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-process",
        sagemaker_session=sess,
        volume_size_in_gb=30,
    )

    processing_inputs = []
    processing_outputs = [
        sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{bucket}/{base_job_prefix}/data/train"),
        sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{bucket}/{base_job_prefix}/data/test"),
    ]

    step_process = ProcessingStep(
        name="Preprocess",
        processor=sklearn_processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        job_arguments=["--num-samples", train_samples],
        code=os.path.join(os.path.dirname(__file__), "preprocess.py"),
    )

    # Training (XGBoost built-in)
    region_name = region
    container = sagemaker.image_uris.retrieve(framework="xgboost", region=region_name, version="1.7-1")
    estimator = Estimator(
        image_uri=container,
        role=role_arn,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{bucket}/{base_job_prefix}/output",
        sagemaker_session=sess,
        base_job_name=f"{base_job_prefix}-train",
        disable_profiler=True,
        enable_sagemaker_metrics=True,
        hyperparameters={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": max_depth,
            "eta": eta,
            "num_round": num_round,
        },
    )

    train_s3 = step_process.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
    test_s3 = step_process.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri

    step_train = TrainingStep(
        name="TrainXGBoost",
        estimator=estimator,
        inputs={
            "train": TrainingInput(s3_data=train_s3, content_type="text/csv"),
            "validation": TrainingInput(s3_data=test_s3, content_type="text/csv"),
        },
    )

    # Model metrics from training job
    metrics = ModelMetrics(
        model_statistics=ModelMetricsSource(
            s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_type="application/json",
        )
    )

    # Register model
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{base_job_prefix}-package-group",
        model_metrics=metrics,
    )

    # Batch transform
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=transform_instance_type,
        output_path=f"s3://{bucket}/{base_job_prefix}/batch-output",
    )
    step_transform = TransformStep(
        name="BatchTransform",
        transformer=transformer,
        inputs=TransformInput(data=test_s3, content_type="text/csv"),
    )

    # Simple quality gate placeholder (would normally parse metrics)
    cond = ConditionGreaterThan(left=0.7, right=0.5)
    step_cond = ConditionStep(name="QualityGate", conditions=[cond], if_steps=[step_register, step_transform], else_steps=[])

    pipeline = Pipeline(
        name=f"{base_job_prefix}-pipeline",
        parameters=[
            processing_instance_type,
            training_instance_type,
            transform_instance_type,
            train_samples,
            max_depth,
            eta,
            num_round,
        ],
        steps=[step_process, step_train, step_cond],
        sagemaker_session=sess,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default=os.environ.get('AWS_REGION', 'us-west-2'))
    parser.add_argument('--role-arn', required=False, default=os.environ.get('SAGEMAKER_ROLE_ARN'))
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--action', choices=['create', 'run'], default='run')
    args = parser.parse_args()

    if not args.role_arn:
        sm = boto3.client('sagemaker', region_name=args.region)
        sts = boto3.client('sts', region_name=args.region)
        account = sts.get_caller_identity()['Account']
        args.role_arn = f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    pipe = build_pipeline(region=args.region, role_arn=args.role_arn, bucket=args.bucket)
    pipe.upsert(role_arn=args.role_arn)

    if args.action == 'run':
        exec_ = pipe.start()
        print(f"Started pipeline execution: {exec_.arn}")
    else:
        print("Pipeline created/updated. Use --action run to start.")


if __name__ == '__main__':
    main()


