# %%
# ################
# ################
# Main experiment.
# ################
# ################
import numpy as np

from reports.util import (
    ModelType,
    extract_success,
    main_preprocess,
    vit_full,
    vit_full_float16,
    vit_full_bfloat_no_threshold,
    resnet_bit,
    resnet_bit_float16,
    resnet_bit_bfloat_no_threshold,
    efficientnet_bit,
    efficientnet_bit_float16,
    efficientnet_bit_bfloat_no_threshold,
)


def main_experiment():
    results = {
        "float32": {
            "vit": main_preprocess(extract_success(vit_full, ModelType.VIT_B_32)),
            "efficient": main_preprocess(
                extract_success(efficientnet_bit, ModelType.EFFICIENTNET)
            ),
            "resnet": main_preprocess(extract_success(resnet_bit, ModelType.RESNET18)),
        },
        "float16": {
            "vit": main_preprocess(
                extract_success(vit_full_float16, ModelType.VIT_B_32)
            ),
            "efficient": main_preprocess(
                extract_success(efficientnet_bit_float16, ModelType.EFFICIENTNET)
            ),
            "resnet": main_preprocess(
                extract_success(resnet_bit_float16, ModelType.RESNET18)
            ),
        },
        "bfloat16": {
            "vit": main_preprocess(
                extract_success(vit_full_bfloat_no_threshold, ModelType.VIT_B_32)
            ),
            "efficient": main_preprocess(
                extract_success(
                    efficientnet_bit_bfloat_no_threshold, ModelType.EFFICIENTNET
                )
            ),
            "resnet": main_preprocess(
                extract_success(resnet_bit_bfloat_no_threshold, ModelType.RESNET18)
            ),
        },
    }

    results = {"float32": results["float32"]}

    # Collect sorted keys for consistent ordering
    dtypes = sorted(results.keys())
    model_types = sorted({mt for dt in dtypes for mt in results[dt].keys()})
    platforms = sorted(
        {
            pl
            for dt in dtypes
            for mt in results[dt].keys()
            if results[dt][mt] is not None
            for pl in results[dt][mt].keys()
        }
    )

    # Begin LaTeX table
    latex = []
    latex.append("\\begin{tabular}{l" + "c" * (len(dtypes) * len(model_types)) + "}")
    latex.append("\\toprule")

    # Header row 1: dtypes spanning their model_type columns
    header_dtype = [""]  # empty top-left cell
    for dt in dtypes:
        header_dtype.append(
            "\\multicolumn{" + str(len(model_types)) + "}{c}{" + dt + "}"
        )
    latex.append(" & ".join(header_dtype) + " \\\\")

    # Add cmidrule under each dtype group
    cmidrules = ["\\cmidrule(lr){1-1}"]  # underline first column
    start = 2
    for _ in dtypes:
        end = start + len(model_types) - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    latex.extend(cmidrules)

    # Header row 2: model types
    header_model = [""]  # empty for first column
    for dt in dtypes:
        for mt in model_types:
            header_model.append(mt)
    latex.append(" & ".join(header_model) + " \\\\")
    latex.append("\\midrule")

    # Table body
    for pl in platforms:
        row = [pl.replace("_", " ")]  # remove underscores for display
        for dt in dtypes:
            for mt in model_types:
                model_data = results[dt].get(mt, None)
                if model_data is not None:
                    cell_value = model_data.get(pl, None)
                    if cell_value is not None:
                        mean, std = cell_value
                        mean_pct = mean * 100
                        std_pct = std * 100
                        cell = f"{mean_pct:.2f}\\% $\\pm$ {std_pct:.2f}\\%"
                    else:
                        cell = ""  # empty for missing platform
                else:
                    cell = ""  # empty for missing model
                row.append(cell)
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    # Print LaTeX table
    print("\n".join(latex))


main_experiment()
