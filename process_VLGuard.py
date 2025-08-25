import json
data_path = './data/VLGuard/train.json'
save_path = './data/VLGuard/VLGuard_train_processed.json'

process_data = []
data = []
with open(data_path, 'r') as data_file:
    data = json.load(data_file)
for i in range(len(data)):
    image_id = data[i]["id"]
    # id 中设置safe safe_instruction为 _ss，safe unsafe_instruction为 _su，unsafe instruction为 _u
    image_name = data[i]["image"]
    if data[i]["safe"]:
        safe_instruction = data[i]["instr-resp"][0]["safe_instruction"]
        safe_response = data[i]["instr-resp"][0]["response"]
        unsafe_instruction = data[i]["instr-resp"][1]["unsafe_instruction"]
        unsafe_response = data[i]["instr-resp"][1]["response"]
        temp_dict_safe = {
                "id":image_id+"_ss",
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<image>\n' + safe_instruction
                    },
                    {
                        "from": "gpt",
                        "value": safe_response
                    }
                ]
            }
        process_data.append(temp_dict_safe)
        temp_dict_unsafe = {
                "id":image_id+"_us",
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<image>\n' + unsafe_instruction
                    },
                    {
                        "from": "gpt",
                        "value": unsafe_response
                    }
                ]
            }
        process_data.append(temp_dict_unsafe)
    else:
        instruction = data[i]["instr-resp"][0]["instruction"]
        response = data[i]["instr-resp"][0]["response"]
        temp_dict = {
                "id":image_id+"_u",
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": '<image>\n' + instruction
                    },
                    {
                        "from": "gpt",
                        "value": response
                    }
                ]
            }
        process_data.append(temp_dict)

ith open(save_path, 'w') as f:
    json.dump(process_data, f, indent=4)