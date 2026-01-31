# %%


# %%

from reports.util import ModelType, extract_success

vit_conv_layer_only_none_stripped = "2025-12-27T19:03:24.958853_ugly-beetle"
vit_conv_layer_only_bit_stripped = "2025-12-27T17:16:57.717541_wet-falcon"
vit_conv_layer_only_full_stripped = "2025-12-29T11:56:09.899807_juicy-anaconda"

vit_conv_layer_only_none_stripped_bfloat16 = "2025-12-28T01:26:10.211169_fine-moth"
vit_conv_layer_only_bit_stripped_bfloat16 = "2025-12-28T11:39:00.661946_ugly-snail"
vit_conv_layer_only_full_stripped_bfloat16 = "2025-12-29T01:09:18.497865_secret-oyster"

vit_conv_layer_only_none_stripped_float16 = "2025-12-28T22:49:26.765736_rich-turtle"
vit_conv_layer_only_bit_stripped_float16 = "2025-12-28T15:04:33.304449_grumpy-bear"
vit_conv_layer_only_full_stripped_float16 = "2025-12-29T06:55:02.664675_slow-bear"


def main_testing_experiment():
    vit0 = extract_success(vit_conv_layer_only_none_stripped, ModelType.VIT_B_32)
    vit1 = extract_success(vit_conv_layer_only_bit_stripped, ModelType.VIT_B_32)
    vit2 = extract_success(vit_conv_layer_only_full_stripped, ModelType.VIT_B_32)

    vit10 = extract_success(
        vit_conv_layer_only_none_stripped_bfloat16, ModelType.VIT_B_32
    )
    vit11 = extract_success(
        vit_conv_layer_only_bit_stripped_bfloat16, ModelType.VIT_B_32
    )
    vit12 = extract_success(
        vit_conv_layer_only_full_stripped_bfloat16, ModelType.VIT_B_32
    )

    vit20 = extract_success(
        vit_conv_layer_only_none_stripped_float16, ModelType.VIT_B_32
    )
    vit21 = extract_success(
        vit_conv_layer_only_bit_stripped_float16, ModelType.VIT_B_32
    )
    vit22 = extract_success(
        vit_conv_layer_only_full_stripped_float16, ModelType.VIT_B_32
    )

    print("bfloat16, none", vit10)
    print("float16, none", vit20)
    print("float32, none", vit0)
    print()
    print("bfloat16, bit", vit11)
    print("float16, bit", vit21)
    print("float32, bit", vit1)
    print()
    print("bfloat16, full", vit12)
    print("float16, full", vit22)
    print("float32, full", vit2)


main_testing_experiment()
