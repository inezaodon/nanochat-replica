"""Alpaca-style prompt formatting (matches Lab 05 SFT template)."""


def format_alpaca_entry(instruction: str, input_text: str | None, response: str) -> tuple[str, str]:
    """
    Build (prompt, response) strings for SFT.

    prompt: everything up to and including "### Response:\\n"
    response: assistant text only (EOS added by caller after tokenization if desired).
    """
    inp = (input_text or "").strip()
    prompt = (
        f"### Instruction:\n{instruction.strip()}\n\n### Input:\n{inp}\n\n### Response:\n"
    )
    return prompt, response.strip()
