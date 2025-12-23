---
description: Goal-Guru uygulamasını başlatma workflow'u
---

# Goal-Guru Uygulaması Başlatma

// turbo-all

## Adımlar

### 1. Klasöre Git
```powershell
cd c:\Users\LENOVO\OneDrive\Masaüstü\Goal-Guru
```

### 2. Bağımlılıkları Yükle (İlk kez)
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Uygulamayı Başlat
```powershell
$env:DATABASE_URL='sqlite:///goal_guru.db'; .\.venv\Scripts\python.exe main.py
```

### 4. Tarayıcıda Aç
- http://127.0.0.1:8080/ adresine git

## Notlar
- Uygulama port 8080'de çalışır
- Virtual environment: `.venv\Scripts\python.exe`
- DATABASE_URL ayarlanmalı (SQLite kullanılıyor)
- API Key: `2f0c06f149e51424f4c9be24eb70cb8f`

## Notlar
- Uygulama port 8080'de çalışır
- Virtual environment path: `.venv\Scripts\python.exe`
- API Key: `2f0c06f149e51424f4c9be24eb70cb8f`
