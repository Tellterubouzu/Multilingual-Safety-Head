import math
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize(values):
    min_v, max_v = min(values), max(values)
    if math.isclose(min_v, max_v):
        # 全ての値が同じなら一律 0.5
        return [0.5] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]

def score_to_color(score):
    r = 255
    g = int(255 * (1 - score))
    b = int(255 * (1 - score))
    return (r, g, b)

def create_attention_html(
    tokens, 
    scores, 
    html_path="attention_visualization.html", 
    max_tokens_per_line=10
):
    html_content = []
    html_content.append("<html>")
    html_content.append("<head>")
    html_content.append("<meta charset='utf-8'>")
    html_content.append("""
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      table { border-collapse: collapse; margin: auto; }
      td { padding: 8px; text-align: center; }
      .token-cell {
        border: 1px solid #ccc;
      }
    </style>
    """)
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append("<table>")
    html_content.append("<tr>")

    current_line_tokens = 0
    for token, score in zip(tokens, scores):
        r, g, b = score_to_color(score)
        color_str = f"rgb({r}, {g}, {b})"
        if current_line_tokens >= max_tokens_per_line:
            html_content.append("</tr><tr>")
            current_line_tokens = 0

        cell_html = (
            f"<td class='token-cell' style='background-color:{color_str};'>"
            f"{token}"
            "</td>"
        )
        html_content.append(cell_html)

        current_line_tokens += 1

    html_content.append("</tr>")
    html_content.append("</table>")
    html_content.append("</body>")
    html_content.append("</html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))
    print(f"Saved attention HTML to: {html_path}")

def group_subwords_to_words(tokens):
    current_word = ""
    for t in tokens:
        if t.startswith("##"):
            sub = t[2:]
            current_word += sub
        else:
            if current_word:
                words.append(current_word)
            current_word = t
    if current_word:
        words.append(current_word)
    return words

def visualize_attention_as_image_excluding_bos_eos(
    model_name: str,
    text: str,
    layer_index: int = -1,
    head_index: int = 0,
    query_token_index: int = None,
    font_path: str = None,
    font_size: int = 20,
    html_path: str = None,
    max_line_chars: int = 100
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager",  # 2023/9月時点 GPT-Neo系など一部モデル用パラメータ
    )
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    attention_matrix = attentions[layer_index][0, head_index]

    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    seq_len = len(tokens)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    exclude_indices = set()
    for idx, tid in enumerate(input_ids):
        if bos_id is not None and tid == bos_id:
            exclude_indices.add(idx)
        if eos_id is not None and tid == eos_id:
            exclude_indices.add(idx)

    valid_indices = [i for i in range(seq_len) if i not in exclude_indices]
    filtered_attention_matrix = attention_matrix[valid_indices][:, valid_indices]
    filtered_tokens = [tokens[i] for i in valid_indices]
    if query_token_index is None:
        attention_scores = filtered_attention_matrix.mean(dim=0).cpu().numpy()
    else:
        attention_scores = filtered_attention_matrix[query_token_index].cpu().numpy()

    norm_scores = normalize(attention_scores)

    if html_path is not None:
        create_attention_html(
            tokens=filtered_tokens,
            scores=norm_scores,
            html_path=html_path,
            max_tokens_per_line=max_line_chars
        )


def compute_attention_by_gradient(
    model_name: str,
    text: str,
    html_path: str = "gradient_attention_demo.html",
    max_line_chars: int = 80,
    do_group_subwords: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.train()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    embedding_grad = model.get_input_embeddings().weight.grad  # [vocab_size, hidden_dim]
    seq_token_ids = input_ids[0]  # (batch_size=1前提)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    # BOS/EOS は可視化対象から除外
    exclude_indices = set()
    for idx, tid in enumerate(seq_token_ids):
        if bos_id is not None and tid.item() == bos_id:
            exclude_indices.add(idx)
        if eos_id is not None and tid.item() == eos_id:
            exclude_indices.add(idx)

    token_grads = []
    tokens_for_display = []

    all_tokens = tokenizer.convert_ids_to_tokens(seq_token_ids)
    for i, tid in enumerate(seq_token_ids):
        if i in exclude_indices:
            continue
        grad_vec = embedding_grad[tid]
        grad_norm = grad_vec.norm(p=2).item()

        token_grads.append(grad_norm)
        tokens_for_display.append(all_tokens[i])

    if do_group_subwords:
        words = group_subwords_to_words(tokens_for_display)
        grouped_grads = []
        tmp_sum = 0.0
        subword_count = 0
        idx_word = 0
        current_prefix = None
        for t, g in zip(tokens_for_display, token_grads):
            if t.startswith("##"):
                tmp_sum += g
                subword_count += 1
            else:
                if subword_count > 0:
                    grouped_grads[-1] = grouped_grads[-1] + tmp_sum
                else:
                    pass
                grouped_grads.append(g)
                tmp_sum = 0.0
                subword_count = 0
        if subword_count > 0:
            grouped_grads[-1] = grouped_grads[-1] + tmp_sum
        token_grads = grouped_grads
        tokens_for_display = words
    norm_scores = normalize(token_grads)
    create_attention_html(
        tokens=tokens_for_display,
        scores=norm_scores,
        html_path=html_path,
        max_tokens_per_line=max_line_chars
    )
    print("Gradient-based attention visualization done.")

if __name__ == "__main__":
    text = (
        "Please explain about Kumamoto prefecture.\nKumamoto is one of the city of Japan. It is famous for Mt.Aso and Kumamoto Castle."
    )
    compute_attention_by_gradient(
        model_name=model_name,
        text=text,
        html_path="gradient_attention_demo.html",
        max_line_chars=20,
        do_group_subwords=True
    )