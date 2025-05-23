# localLLM — NPC Dialogue Engine

A C++ project for advanced, real-time non-player character (NPC) dialogue powered by a local large language model (LLM). Features dynamic, personality-driven responses with cutting-edge Zipfian token optimization.

---

## Features

- **Enable MMAP, and GPU-Layers if Applicable**

- **Real-Time, In-Character NPC Dialogue**  
  NPCs respond contextually with personalities (friendly, rude, suspicious, etc.) using local LLM inference.

- **Advanced Token Sampling**  
  Integrates `zipf.h` — a ZipfAccelerator class for logit biasing, penalizing repetition, and dynamically tuning dialogue quality.

- **Cross-Platform**  
  Runs on both Linux and Windows, using [llama.cpp](https://github.com/ggerganov/llama.cpp) as the backend.

- **Configurable Personalities**  
  Easily expand to new NPC types and moods by editing C++ structs.

---

## Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/picsyy/localLLM.git
    cd localLLM
    ```

2. **Install Dependencies**
    - C++17 compatible compiler (g++ recommended)
    - [llama.cpp](https://github.com/ggerganov/llama.cpp) (follow their build instructions)
    - (Optional) SDL2 for advanced rendering

3. **Get Model Weights**
    - Download `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1).
    - Place it in a `model/` folder at the repo root:
      ```
      localLLM/
        └── model/
            └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
      ```
    - **Note:** Model weights are not distributed in this repository.

4. **Build and Run**
    ```bash
    g++ -O2 -std=c++17 rolled.cpp -o npc_dialogue -Iinclude -Llibs -lllama
    ./npc_dialogue
    ```

---

## How It Works

- **rolled.cpp**  
  The main engine: handles NPC personality, prompt construction, dialogue flow, token sampling, and overall interaction logic.

- **zipf.h**  
  Implements ZipfAccelerator, which precomputes token categories (common, rare, punctuation, etc.), applies logit biasing, and provides repetition penalty logic for high-quality, natural output.

---

## Example Usage

Interact with multiple NPCs:
Choose NPC to converse with:
0: Krackle - deadly front door guard...
1: Mira - world-weary tavernkeeper...
2: Feylan - anxious court scribe...
Enter NPC number: 0
You: May I enter the castle?
Krackle: "State your business. The dynasty doesn't tolerate idlers."


---

## Project Structure

- `rolled.cpp` — Main program logic
- `zipf.h` — Zipfian logit optimization and token management
- `README.md` — This file
- `model/` — Place your downloaded LLM weights here

---

## License

MIT License

---

## Author

Damen Garland  
DamenGarland@gmail.com

---

## Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) (LLM backend)
- Zipfian token biasing and dynamic dialogue system: Damen Garland

---
