import os
import re
import json
import argparse


OPTION_ENTRY_PATTERN = r'^([A-Z])\.\s*(.+)$'
def extract_option_and_answer(option_text):
    match = re.match(OPTION_ENTRY_PATTERN, option_text.strip())
    if match:
        option = match.group(1)
        answer = match.group(2).strip()
        return option, answer
    else:
        raise ValueError(f"Invalid option text: {option_text}")

def extract_correct_answer(options, correct_answer):
    for option in options:
        option, answer = extract_option_and_answer(option)
        if option == correct_answer:
            return answer
    print(f"Correct answer not found in options: {correct_answer}. Expected answer: {correct_answer}")
    return None

def build_dataset_entry(original_entry, question_number):
    correct_answer = extract_correct_answer(original_entry['options'], original_entry['answer'])
    if correct_answer is None:
        print(f"Correct answer not found with video ID: {original_entry['videoID']}! Skipping...")
        return None
    return {
        'question': original_entry['question'],
        'answer': correct_answer,
        'question_id': f'{original_entry['videoID']}_Q{question_number:02d}'
    }

def align_dataset(original_dataset_dir, video_dir, mem_dir, output_dir, duration_filter = ['long']):
    aligned_dataset = {}
    with open(original_dataset_dir, 'r') as f:
        for line in f:
            qa_entry = json.loads(line)
            if qa_entry['duration'] not in duration_filter:
                continue
            video_id = qa_entry['videoID']
            if video_id not in aligned_dataset.keys():
                aligned_dataset[video_id] = {
                    'video_path': os.path.join(video_dir, f'{video_id}.mp4'),
                    'mem_path': os.path.join(mem_dir, f'{video_id}.pkl'),
                    'qa_list': []
                }
            qa_list = aligned_dataset[video_id]['qa_list']
            aligned_entry = build_dataset_entry(qa_entry, len(qa_list) + 1)
            if aligned_entry is not None:
                qa_list.append(aligned_entry)
    with open(output_dir, 'w') as f:
        json.dump(aligned_dataset, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dataset_dir', type=str, default='./videomme/test-00000-of-00001.jsonl')
    parser.add_argument('--video_dir', type=str, default='/home/hk-project-p0022573/lmu_xjh4853/workspace_ysk_wht/hkfswork/lmu_xjh4853-m3-agent/m3-agent-videomme/data/videos')
    parser.add_argument('--mem_dir', type=str, default='/home/hk-project-p0022573/lmu_xjh4853/workspace_ysk_wht/hkfswork/lmu_xjh4853-m3-agent/m3-agent-videomme/data/memory_graphs')
    parser.add_argument('--duration_filter', nargs='+', default=['long'])
    parser.add_argument('--output_dir', type=str, default='./videomme/videomme.json')
    args = parser.parse_args()
    align_dataset(args.original_dataset_dir, args.video_dir, args.mem_dir, args.output_dir, args.duration_filter)
