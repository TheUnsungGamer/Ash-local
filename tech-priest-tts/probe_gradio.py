from gradio_client import Client

c = Client("http://127.0.0.1:7897/")

for i in range(10):
    try:
        print(f"\nTrying fn_index={i}")
        result = c.predict(r"C:\Users\richa\test.wav", fn_index=i)
        print("SUCCESS:", result)
    except Exception as e:
        print("FAIL:", str(e)[:200])