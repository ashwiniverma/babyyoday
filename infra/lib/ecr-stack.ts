import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export class EcrStack extends cdk.Stack {
  readonly repository: ecr.Repository;

  constructor(scope: Construct, id: string, props: cdk.StackProps) {
    super(scope, id, props);

    this.repository = new ecr.Repository(this, "AgentRepo", {
      repositoryName: "babyyoday-agent",
      imageScanOnPush: true,
      encryption: ecr.RepositoryEncryption.AES_256,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          // Remove untagged images after 7 days
          maxImageAge: cdk.Duration.days(7),
          tagStatus: ecr.TagStatus.UNTAGGED,
          rulePriority: 1,
        },
        {
          // Keep only the last 10 images — ANY must have highest priority
          maxImageCount: 10,
          tagStatus: ecr.TagStatus.ANY,
          rulePriority: 2,
        },
      ],
    });

    new cdk.CfnOutput(this, "RepositoryUri", { value: this.repository.repositoryUri });
  }
}
