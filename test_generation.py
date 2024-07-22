import torch
from refiners.foundationals.latent_diffusion.stable_diffusion_1.ella_adapter import SD1ELLAAdapter
from refiners.foundationals.latent_diffusion import StableDiffusion_1
from refiners.fluxion.utils import manual_seed, no_grad
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn as nn
from refiners.fluxion.utils import load_from_safetensors

test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class T5TextEmbedder(nn.Module):
    def __init__(self, pretrained_path="google/flan-t5-xl", max_length=None):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = max_length

    def forward(self, caption, text_input_ids=None, attention_mask=None, max_length=None):
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(caption, return_tensors="pt", add_special_tokens=True)
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings


@no_grad()
def main():
    sd15 = StableDiffusion_1(device=test_device, dtype=dtype)
    sd15.clip_text_encoder.load_from_safetensors("tests/weights/CLIPTextEncoderL.safetensors")
    sd15.lda.load_from_safetensors("tests/weights/lda.safetensors")
    sd15.unet.load_from_safetensors("tests/weights/unet.safetensors")
    t5_encoder = T5TextEmbedder().to(test_device, dtype=dtype)
    prompt = "Crocodile in a sweater"
    llm_text_embedding = t5_encoder(prompt, max_length=128).to(test_device, dtype=dtype)

    # negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt)

    adapter = SD1ELLAAdapter(
        target=sd15.unet, weights=load_from_safetensors("tests/weights/ELLA-Adapter/ella-sd1.5-tsc-t5xl.safetensors")
    )
    # adapter.inject()
    adapter.set_llm_text_embedding(llm_text_embedding)

    sd15.set_inference_steps(20)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype = sd15.dtype)

    for step in sd15.steps:
        # adapter.set_llm_text_embedding(llm_text_embedding)
        print(step)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)
    predicted_image.save("output_ella.png")


if __name__ == "__main__":
    main()
