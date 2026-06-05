import subprocess

token_code = "4/0AeoWuM8DfZ4YGTX-1ZNZkZZr7xCuUlZvkVioVANtj7-nki54hWn_QWOaaK_KKLFSkm4wPw"

def login_antigravity_cli(token_code):
    try:
        print("Starting agy models...")
        process = subprocess.Popen(
            ["agy", "models"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Communicating token_code...")
        stdout, stderr = process.communicate(input=f"{token_code}\n", timeout=20)
        return process.returncode, stdout, stderr
    except Exception as e:
        return -1, "", str(e)

code, stdout, stderr = login_antigravity_cli(token_code)
print(f"Return Code: {code}")
print(f"Stdout:\n{stdout}")
print(f"Stderr:\n{stderr}")
