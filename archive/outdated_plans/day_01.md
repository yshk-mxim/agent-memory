# Day 1 (Monday): Environment Setup and API Validation

**Week 1 - Day 1**

---

**Objectives:**
- Set up complete development environment
- Verify all API connections work correctly
- Create project directory structure
- Download and test Llama model

**Tasks:**

| Task | Time | Details |
|------|------|---------|
| Install llama-cpp-python with Metal support | 1.5h | `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python` |
| Download Llama-3.1-8B Q4 GGUF model | 1h | Download from HF (~4.9GB) |
| Test Claude API connection | 30m | Simple test call with Haiku model |
| Test DeepSeek R1 API connection | 45m | Use OpenAI-compatible client, verify reasoning traces |
| Test local Llama inference | 1h | Generate simple response, verify Metal acceleration |
| Create utility module | 1.5h | `src/utils.py` with API client wrappers |
| Create requirements.txt | 30m | All dependencies documented |

**Files to Create:**
- `/Users/dev_user/semantic/src/__init__.py`
- `/Users/dev_user/semantic/src/config.py` - Load env.json
- `/Users/dev_user/semantic/src/utils.py` - API client wrappers (Claude, DeepSeek, Llama)
- `/Users/dev_user/semantic/requirements.txt`

**Success Criteria:**
- [ ] Can load Llama-3.1-8B Q4 GGUF and generate response in <30 seconds
- [ ] Claude API returns valid response
- [ ] DeepSeek R1 API returns response with reasoning_content field
- [ ] All APIs callable from unified interface in utils.py

**DeepSeek R1 API Pattern:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="<from env.json>",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": prompt}]
)
# Access: response.choices[0].message.content
```

**Llama GGUF Pattern:**
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1  # Use Metal on Mac
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)
```

---

---

## Quick Reference

**Previous Day:** [Day 0](day_00.md) (if exists)
**Next Day:** [Day 2](day_02.md) (if exists)
**Complete Plan:** [Complete 3-Week Plan](../complete_plan.md)

---

## Checklist for Today

- [ ] Review objectives and tasks
- [ ] Set up required files and dependencies
- [ ] Execute all tasks according to timeline
- [ ] Verify success criteria
- [ ] Document any issues or deviations
- [ ] Prepare for next day

---

*Generated from complete_plan.md*
