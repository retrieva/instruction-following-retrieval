from ..dataset.msmarco import MSMARCO
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity
from tqdm import tqdm
from datasets import load_dataset



def main():
    msmarco = load_dataset("InF-IR/InF-IR")["msmarco"]
    examples = [i for i in msmarco]

    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    batch_size = 32
    
    scores = []
    id = 0
    for i in tqdm(range(0, len(examples), batch_size)):
        batch_examples = examples[i:i + batch_size]

        all_texts = []
        for example in batch_examples:
            query = example["query_positive"]

            instruction_positive = example["instruction_positive"]
            instruction_negative = example["instruction_negative"]

            x_positive = example["instruction_positive"] + example["query_positive"]
            x_negative = example["instruction_negative"] + example["query_positive"]

            passage_positive = example["document_positive"]
            passage_negative = example["hard_negative_document_1"]

            all_texts.extend([
                query,
                instruction_positive,
                instruction_negative,
                x_positive,
                x_negative,
                passage_positive,
                passage_negative,
            ])

        doc_vecs = angle.encode(all_texts, normalize_embedding=True)

        for j, example in enumerate(batch_examples):
            
            start_idx = j * 7
            end_idx = start_idx + 7
            sample_vecs = doc_vecs[start_idx:end_idx]

            q_and_p_pos_score = cosine_similarity(sample_vecs[0], sample_vecs[5]) # クエリと正例文書
            q_and_p_neg_score = cosine_similarity(sample_vecs[0], sample_vecs[6]) # クエリと負例文書

            inst_pos_and_p_pos_score = cosine_similarity(sample_vecs[1], sample_vecs[5]) # 正例指示文 と 正例文書
            inst_pos_and_p_neg_score = cosine_similarity(sample_vecs[1], sample_vecs[6]) # 正例指示文 と 負例文書

            inst_neg_and_p_pos_score = cosine_similarity(sample_vecs[2], sample_vecs[5]) # 負例指示文 と 正例文書
            inst_neg_and_p_neg_score = cosine_similarity(sample_vecs[2], sample_vecs[6]) # 負例指示文 と 負例文書

            x_pos_and_p_pos_score = cosine_similarity(sample_vecs[3], sample_vecs[5]) # クエリ+正例指示文 と 正例文書
            x_pos_and_p_neg_score = cosine_similarity(sample_vecs[3], sample_vecs[6]) # クエリ+正例指示文 と 負例文書

            x_neg_and_p_pos_score = cosine_similarity(sample_vecs[4], sample_vecs[5]) # クエリ+負例指示文 と 正例文書
            x_neg_and_p_neg_score = cosine_similarity(sample_vecs[4], sample_vecs[6]) # クエリ+負例指示文 と 負例文書

            scores.append({
                "id": id,
                "q_and_p_pos_score": float(q_and_p_pos_score),
                "q_and_p_neg_score": float(q_and_p_neg_score),
                "inst_pos_and_p_pos_score": float(inst_pos_and_p_pos_score),
                "inst_pos_and_p_neg_score": float(inst_pos_and_p_neg_score),
                "inst_neg_and_p_pos_score": float(inst_neg_and_p_pos_score),
                "inst_neg_and_p_neg_score": float(inst_neg_and_p_neg_score),
                "x_pos_and_p_pos_score": float(x_pos_and_p_pos_score),
                "x_pos_and_p_neg_score": float(x_pos_and_p_neg_score),
                "x_neg_and_p_pos_score": float(x_neg_and_p_pos_score),
                "x_neg_and_p_neg_score": float(x_neg_and_p_neg_score),
            })
            id += 1
        
    with open("./dataset/similarity/similarity.json", "w") as f:
        import json
        json.dump(scores, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
