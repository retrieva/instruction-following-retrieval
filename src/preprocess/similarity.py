from ..dataset.msmarco import MSMARCO
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity
from tqdm import tqdm



def main():
    msmarco = MSMARCO()
    examples = [i for i in msmarco]

    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    batch_size = 32 
    
    scores = []
    id = 0
    for i in tqdm(range(0, len(examples), batch_size)):
        batch_examples = examples[i:i + batch_size]

        all_texts = []
        for example in batch_examples:
            passage_positive = example.texts[1]
            passage_negative = example.texts[2]
            x_positive = example.texts[3]
            
            all_texts.extend([
                x_positive,
                passage_positive,
                passage_negative,
            ])

        doc_vecs = angle.encode(all_texts, normalize_embedding=True)
        
        for j, example in enumerate(batch_examples):
            start_idx = j * 3
            end_idx = start_idx + 3
            sample_vecs = doc_vecs[start_idx:end_idx]
            positive_score = cosine_similarity(sample_vecs[0], sample_vecs[1]) # x_positive と passage_positive
            negative_score = cosine_similarity(sample_vecs[0], sample_vecs[2]) # x_positive と passage_negative
            scores.append({"id": id, "positive_score": float(positive_score), "negative_score": float(negative_score)})
            id += 1

    with open("/data/sugiyama/dataset/similarity/similarity.json", "w") as f:
        import json
        json.dump(scores, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()