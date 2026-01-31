# %%

from reports.util import (
    ModelType,
    extract_success,
    main_preprocess,
    resnet_use_deterministic,
)


print(
    main_preprocess(
        extract_success(resnet_use_deterministic, model_type=ModelType.RESNET18)
    )
)
