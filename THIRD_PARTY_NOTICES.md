# Attribution and license scope

Original orchestration and documentation in this recipe are MIT licensed. This grant does not relicense model weights, upstream images, kernels, or patches derived from upstream code.

- **Altar-1 and GLM-5.3**: The model checkpoint is downloaded separately and remains subject to its [model card](https://huggingface.co/AikidoSec/altar-1) and [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE). This repository does not redistribute the weights.
- **SparkRing**: The pinned base image and switchless NCCL transport retain their upstream licenses. See [SparkRing source](https://github.com/FujitsuPolycom/sparkring) and the retained [NCCL notice](licenses/SparkRing-NOTICE.txt).
- **vLLM and B12X**: The patches in `patches/` and `image/` modify upstream Apache-2.0 code and retain upstream copyright and license terms. See [Apache-2.0](licenses/Apache-2.0.txt), [vLLM source](https://github.com/local-inference-lab/vllm), [B12X source](https://github.com/local-inference-lab/b12x), and the notices alongside the patch stages.

The included patches are local compatibility changes and do not imply upstream endorsement.
