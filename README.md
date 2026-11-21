# OCR-D extension to recognize text using LightOnOCR models

> ocrd_lightonocr_recognizer is an OCR‑D module that tries to recognize text using LightOnOCR models.
       
_Disclaimer_: Work in progeress

## Installation

```commandline
pip install .
```     

_Docker_ version will be available soon:

## Quick Start

```commandline
ocrd-lightonocr-recognizer \
  -I OCR-D-SEG \
  -O OCR-D-SEG-TEXT-RECOGNIZED \
  -p '{"model_id":"lightonai/LightOnOCR-3-1B", "device": "cpu", "level-of-operation": "region"}'
```
