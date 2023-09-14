import transformers
import torch
from transformers import AutoTokenizer
from person.action.system_setting.system1.chat_template import generate_system_message
from person.action.brain.agent import BaseAgent

MODEL_PATH = "/data/ckpts/huggingface/models/models--mosaicml--mpt-30b-chat/snapshots/54f33278a04aa4e612bca482b82f801ab658e890"


class MPT(BaseAgent):
    def __init__(self,profile_version, system_version, model_name="mpt-30b-chat", person_name="monica", model_path=MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        config = transformers.AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # config.max_seq_len = 16384 # (input + output) tokens can now be up to 16384
        # config.init_device="meta"
        config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
        config.init_device = 'cuda'  # For fast initialization directly on GPU!

        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model_path = model_path

        super().__init__(person_name=person_name, model=model, model_name=model_name, profile_version=profile_version, system_version=system_version)

        self.system_message = F"[INST] <<SYS>>\n{generate_system_message(person_name)[0].content}\n<</SYS>>\n"

    def run(self, user_input):
        user_input = self.system_message + user_input + "\n[/INST]\n"
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(user_input, return_tensors="pt").to('cuda')
            outputs = self.chat_model.generate(**inputs,
                                               temperature=0.01,
                                               max_length=8000,
                                               num_return_sequences=1,
                                               max_new_tokens=50,
                                               early_stopping=True,
                                               use_cache=True,
                                               top_k=1
                                               )

            result2 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            first_end_ins = result2.find('[/INST]')
            result2 = result2[first_end_ins + 8:]
            second_end = result2.find('---')
            result2 = result2[:second_end].strip()
        return result2

    def clear(self):
        self.chat_history = []


if __name__ == "__main__":
    mpt = MPT()
    print(mpt.run("who are you?"))
