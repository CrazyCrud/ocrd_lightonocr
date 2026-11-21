import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor, LogitsProcessor, LogitsProcessorList


class WindowRepetitionPenalty(LogitsProcessor):
    def __init__(self, penalty=1.1, window=20):
        self.penalty, self.window = penalty, window

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = input_ids[b, -self.window:].unique()
            s = scores[b, ids]
            scores[b, ids] = torch.where(s < 0, s * self.penalty, s / self.penalty)
        return scores

