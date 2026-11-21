from ocrd import Processor
from ocrd.processor.base import OcrdPageResult
from ocrd_models.ocrd_page import (
    OcrdPage as PcGtsType, TextEquivType
)
from PIL import Image
import base64
import io
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor, LogitsProcessorList
from .helpers import WindowRepetitionPenalty 


class OcrdLightOnOcrRecognizer(Processor):
    @property
    def executable(self) -> str:
        return "ocrd-lightonocr-recognizer"

    def setup(self):
        self.params = dict(self.parameter or {})

        self.device = "cpu" if self.params['device'] == "cpu" else "cuda"
        model_id = self.params['model_id']

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        self.processor = LightOnOcrProcessor.from_pretrained(model_id)

        self.operation_level = self.params['level-of-operation']
        
    def process_page_pcgts(self, *input_pcgts: PcGtsType, page_id: str | None = None) -> OcrdPageResult:
        """
        input_pcgts[0]: PageXML with textregions (or textlines)
        Return a new PageXML
        """

        assert len(input_pcgts) == 1, "Expect exactly 1 input PageXML file"
        lines_doc = input_pcgts[0]

        output_doc: PcGtsType = self.recognize_text(lines_doc, page_id=page_id or "")

        return OcrdPageResult(pcgts=output_doc)

    def recognize_text(self, input_doc: PcGtsType, page_id: str):
        """
        Main function.
        """

        # Create output document from the input PageXML document.
        out_doc = input_doc
        out_page = out_doc.get_Page()

        page_image, page_coords, _ = self.workspace.image_from_page(out_page, page_id)

        all_regions = out_page.get_AllRegions(classes=["Text"])
        print(f"{page_id} has {len(all_regions)} TextRegion elements")

        # Crop TextRegion elements.
        for text_region in all_regions:           
            print(f"Crop TextRegion {text_region.id}")
            text_region_image, text_region_coords = self.workspace.image_from_segment(
                text_region, page_image, page_coords
            )

            if self.operation_level == "region":
                text_region_image_base64 = self.image_to_base64(text_region_image)

                generated_text = self.predict_text(text_region_image_base64)
                
                if text_region.get_TextEquiv():
                    text_region.set_TextEquiv([])
                text_region.add_TextEquiv(TextEquivType(Unicode=generated_text))
            else:
                lines = text_region.get_TextLine() or []
                if len(lines) < 1:
                    continue
                
                for line in lines:
                    try:
                        print(f"Crop TextLine {line.id}")
                        line_image, _ = self.workspace.image_from_segment(
                            line, text_region_image, text_region_coords
                        )

                        generated_text = self.predict_text(line_image)
                        if line.get_TextEquiv():
                            line.set_TextEquiv([])
                        line.add_TextEquiv(TextEquivType(Unicode=generated_text))

                    except Exception as e:
                        print(f"Error while croping images {e}")
                        continue
        
        return out_doc

    def predict_text(self, element_image: Image.Image):
        generated_text = ""
        try:
            # See https://huggingface.co/b-straub/LightOnOCR-3-1B-Historic-German-Handwriting#transformers-%E2%89%A5-50
            conversation = [{"role": "user", "content": [{"type": "image", "url": element_image}]}]
            inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt").to(self.device)
            # See4 https://fmorenovr.medium.com/bfloat16-vs-float16-vs-float32-understanding-tensor-dtypes-in-large-language-models-f36869d964f4
            if self.device == "cpu":
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
            else:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            out = self.model.generate(**inputs, max_new_tokens=7024, do_sample=False,
                                logits_processor=LogitsProcessorList([WindowRepetitionPenalty(1.1, 20)]))
            generated_text = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
        except Exception as e:
            print(f"Error while recognizing text {e}")

        return generated_text

    def image_to_base64(self, img: Image.Image) -> str:
        format_ = img.format or "PNG"  # fallback

        buffer = io.BytesIO()
        img.save(buffer, format=format_, quality=100)
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode('utf-8')

        mime_type = f"image/{format_.lower()}"
        if format_.upper() == "JPEG":
            mime_type = "image/jpeg"              

        return f"data:{mime_type};base64,{base64_str}"