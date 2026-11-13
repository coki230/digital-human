# sync all code to server
rsync -avz --exclude '.*' . coki@192.168.1.24:/home/coki/humanAI/dev1/digital-human/


# 2. 通过 SSH 在远程服务器上执行命令
#ssh coki@192.168.1.24 << 'EOF'
#    cd /home/coki/humanAI/dev1/digital-human/
#    # 加入一个软连接
#    ln -s third_party/flash_head flash_head
#    # 进入项目目录
#    source .venv/bin/activate
#    # 运行你的 Python 程序（如果需要后台运行，后面可以加 &）
#    export TORCH_COMPILE_DISABLE=1
#    # test SoulX-FlashHead
#    python test.py
#EOF

#ssh coki@192.168.1.24 << 'EOF'
#    cd /home/coki/humanAI/dev1/digital-human/
#    # 进入项目目录
#    source .venv/bin/activate
#    # test pipeline.py
##    export HF_ENDPOINT="https://hf-mirror.com"
##    export PYTHONPATH=$PWD
#
#    python src/audio/pipeline.py
#EOF

#ssh coki@192.168.1.24 << 'EOF'
#    cd /home/coki/humanAI/dev1/digital-human/
#    # 进入项目目录
#    source .venv/bin/activate
#    python -m src.avatar.engine
#EOF

ssh coki@192.168.1.24 << 'EOF'
    # 删除上一次生成的音视频
    rm -rf /home/coki/humanAI/dev1/digital-human/outputs/test_videos/*
    cd /home/coki/humanAI/dev1/digital-human/
    # 进入项目目录
    source .venv/bin/activate
#    python -m src.llm.ollama_client
    export HF_ENDPOINT="https://hf-mirror.com"
    export PYTHONPATH=$PWD
    python -m scripts.test_text_to_avatar
EOF
