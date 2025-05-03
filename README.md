# Tutor Week 10
## Cara deploy model ML dengan Fast-API
1. buat model dan/atau tokenizer (misal NLP) dan simpan dalam bentuk (.h5, .pkl, .joblib)
2. install uvicorn dengan pip di terminal `pip install uvicorn`
3. install fast-api denan pip di terminal `pip install fastapi`
4. buat project fastapi dengan contoh struktur project:
   ```
   --app
     |---api
         |---kode.py
     |---model
         |---model.h5
         |---tokenizer.pkl 
   ```
5. dengan kode yang ada pada repo ini, buka terminal dan jalankan pada posisi di direktori root project (di atas folder app), menggunakan kode: `uvicorn app.api.kode:app --reload`
6. fastapi akan berjalan pada http://127.0.0.1:8000/ atau http://localhost:8000/, pergi ke url itu di browser tambahkan `/docs` agar menuju ke swagger untuk test API kita.
7. pergi ke https://dashboard.ngrok.com/ , buat akun dan pergi ke https://dashboard.ngrok.com/get-started/setup/windows dan ikuti cara untuk install dan set-up ngrok di komputer.
8. setelah itu pergi ke tab domain pada dashboard ngrok https://dashboard.ngrok.com/domains disitu tambahkan domain baru agar nantinya saat ngrok dijalankan di komputer,
   link yang digenerate selalu sama (static).
9. setelah ngrok sudah ter-setup dan mendapatkan domain, maka tinggal tunneling atau expose url localhost fast-api tadi ke ngrok, dengan command di terminal:
   `ngrok http --url=nama.domain.kamu.free.app 8000`, nanti kamu akan mendapatkan url tunneling di alamat `https://nama.domain.kamu.free.app/`
10. tinggal dicoba di browser ketik alamat `https://nama.domain.kamu.free.app/docs` untuk mencoba API nya.
