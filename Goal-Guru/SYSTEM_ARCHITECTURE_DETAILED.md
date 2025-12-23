# Football Prediction System - Detaylı Sistem Mimarisi & Hiyerarşik Bağlantılar

## 📋 İçindekiler
1. [Genel Sistem Akışı](#genel-sistem-akışı)
2. [Veri Akışı Diyagramı](#veri-akışı-diyagramı)
3. [Lambda Hesaplaması - Çapraz Sistem](#lambda-hesaplaması---çapraz-sistem)
4. [Kesin Skor Tahmini](#kesin-skor-tahmini)
5. [Tahmin Türleri ve Bağlantıları](#tahmin-türleri-ve-bağlantıları)
6. [Ensemble Kombinasyon Sistemi](#ensemble-kombinasyon-sistemi)
7. [Implementation Detayları](#implementation-detayları)

---

## Genel Sistem Akışı

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FOOTBALL PREDICTION SYSTEM                      │
└─────────────────────────────────────────────────────────────────────┘

ADIM 1: VERİ ALIMI
├─ API'den takım bilgileri çek (Football-Data.org / API-Football)
├─ Son 5-10 maçın detaylarını al
├─ ELO ve form verilerini hazırla
└─ Lig bilgilerini topla (cross-league kontrolü için)

ADIM 2: BAŞLANGIÇ HESAPLAMALAR
├─ XG (Expected Goals) hesapla
├─ XGA (Expected Goals Against) hesapla
├─ Form skoru hesapla (W/L/D analizi)
├─ ELO rating'i hesapla
└─ Ev/Deplasman avantajı analiz et

ADIM 3: LAMBDA HESAPLAMASI (ÇÖP CROSS LAMBDA)
├─ XG ve XGA nilai'ler kullanarak temel lambdayı hesapla
├─ ELO farkını dikkate al
├─ Lig farkını (cross-league) ayarla
├─ Kış/Yaz etkisini uygula
└─ Son 5 maç venue-specific değerlerini ekle
    ↓
    → λ_home ve λ_away çıkışları

ADIM 4: POISSON / DIXON-COLES MATRİSİ
├─ λ_home ve λ_away → Olasılık matrisi dönüştür
├─ Beraberlik koruma mekanizması uygula
├─ Ekstrem maç kontrolü (lambda > 4.0)
└─ Matriski normalize et

ADIM 5: KESIN SKOR ÇIKARIMI
├─ Matristen tüm skor kombinasyonlarını al
├─ En olası skorları sırala (top 5-10)
├─ Skor olasılıklarını yüzde olarak dönüştür
└─ 1X2 sonuçlarıyla tutarlılık kontrol et

ADIM 6: TEMEL TAHMINLER (1X2, BTTS, O/U)
├─ 1X2 tahminleri: home_win, draw, away_win
├─ Over/Under 2.5
├─ BTTS (Both Teams To Score): Yes/No
└─ Expected Goals formatı

ADIM 7: GELIŞMIŞ TAHMINLER
├─ Half-Time/Full-Time
├─ Handicap (+1, -1, +1.5 vb)
├─ Goal Range (1-3 gol, 3+ vb)
├─ Double Chance
├─ Team Goals (hangi takım daha fazla atar)
└─ Correct Score (kesin skorlar detaylı)

ADIM 8: PSİKOLOJİK AYARLAMALAR
├─ Motivasyon farkını hesapla
├─ Momentum analizi yap
├─ Tahminleri %10'a kadar ayarla
└─ Beraberlik minimum %12 sınırını koru

ADIM 9: ENSEMBLE KOMBİNASYONU
├─ Tüm model tahminlerini topla:
│  ├─ Poisson Model
│  ├─ Dixon-Coles Model
│  ├─ XGBoost Model
│  ├─ Hybrid ML System
│  ├─ CRF Predictor
│  ├─ Self-Learning Model
│  └─ Neural Network
├─ Dinamik ağırlık sistemi uygula
├─ Meta-learning layer ile akıllı seçim yap
├─ Lig farkı düzeltmesi ekle (cross-league)
└─ Final tahminler üret

ADIM 10: GÜVEN VE AÇIKLAMA
├─ Güven seviyesi hesapla (45-90%)
├─ Tahmin varyansını analiz et
├─ Şüpheli sonuçlara uyar ekle
└─ Açıklamalar (XAI) oluştur
```

---

## Veri Akışı Diyagramı

```
┌─────────────────────────────────┐
│     API VERİ ALIMI              │
│ • Team Stats                    │
│ • Recent Matches (5-10 games)   │
│ • League Info                   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   BAŞLANGIÇ HESAPLAMALARı       │
├─────────────────────────────────┤
│ Home Team Analysis:             │
│  ├─ XG (expected goals for)    │
│  ├─ XGA (expected goals against)│
│  ├─ Form Score: W/L/D          │
│  ├─ ELO Rating                 │
│  ├─ Home Performance (last 5)  │
│  └─ Venue Bonus                │
│                                 │
│ Away Team Analysis (aynı)      │
└─────────────┬───────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│        LAMBDA HESAPLAMASI                │
│   (calculate_lambda_cross)               │
├──────────────────────────────────────────┤
│ INPUT:                                   │
│  • home_xg, home_xga                    │
│  • away_xg, away_xga                    │
│  • elo_diff                             │
│  • home_team_data, away_team_data       │
│  • match_context (lig, derbi vb)        │
│                                          │
│ HESAPLAMA ADIMLARI:                     │
│ 1. Temel Lambda (xG × 0.876 + G × 0.124)│
│ 2. ELO Adjustment (±5-15%)              │
│ 3. Form Boost (son 5 maç +/- 10%)       │
│ 4. Cross-League Adjustment:             │
│    ├─ UEFA lig ise +20%                 │
│    ├─ Alt lig ise -50%                  │
│    └─ Aynı lig ise ×1.0                 │
│ 5. Venue Bonus (ev 65%, deplasman 35%)  │
│ 6. Rest Days Effect (istirahat günleri) │
│ 7. Derby Factor (derbi ise +/-5%)       │
│                                          │
│ OUTPUT: λ_home, λ_away                  │
└───────────────┬────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
    POISSON MODEL   DIXON-COLES MODEL
    ├─ Normal λ     ├─ Düşük skor düzeltme (0-0, 1-0, 0-1, 1-1)
    └─ Favori boost └─ Rho parametresi (0.05)
        (1.15x)
        │                │
        └────────┬───────┘
                 ▼
        ┌──────────────────┐
        │ OLASILILIK MATRİSİ│
        ├──────────────────┤
        │ 0-0: 8.2%        │
        │ 1-0: 15.3%       │
        │ 0-1: 12.1%       │
        │ 1-1: 18.5%       │
        │ 2-0: 8.7%        │
        │ 2-1: 10.2%       │
        │ ... ve devamı    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │  KESIN SKOR ÇIKARIMI     │
        │  (get_exact_score_...)   │
        ├──────────────────────────┤
        │ Top 5 En Olası Skor:    │
        │ 1. 1-1: 18.5%           │
        │ 2. 1-0: 15.3%           │
        │ 3. 2-1: 10.2%           │
        │ 4. 0-1: 12.1%           │
        │ 5. 2-0: 8.7%            │
        └────────┬────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │   1X2 TAHMİNLERİ ÇIKARIMI│
        ├──────────────────────────┤
        │ Home Win: 40.8%          │
        │ Draw: 35.2%              │
        │ Away Win: 24.0%          │
        │                          │
        │ Beraberlik Kontrol:      │
        │ ├─ Min: 15% (empoze)    │
        │ ├─ Güncel: 35.2%        │
        │ └─ OK ✓                  │
        └────────┬────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │   DİĞER PAZARLAR         │
        ├──────────────────────────┤
        │ Over 2.5: 62.3%          │
        │ Under 2.5: 37.7%         │
        │ BTTS Yes: 51.2%          │
        │ BTTS No: 48.8%           │
        └────────┬────────────────┘
                 │
    ┌────────────┴───────────┐
    ▼                        ▼
HT/FT, Handicap,  Goal Range, Double
Team Goals        Chance vb
    │                        │
    └────────────┬───────────┘
                 │
                 ▼
        ┌──────────────────────────────────┐
        │  PSİKOLOJİK AYARLAMALAR         │
        ├──────────────────────────────────┤
        │ • Motivasyon Farkı: +3%          │
        │ • Momentum: Home Advantage       │
        │ • Adjustment: ±10% max           │
        │ • Draw Floor: %12 minimum        │
        │ • Result:                        │
        │   ├─ Home: 41.8%                 │
        │   ├─ Draw: 35.2%                 │
        │   └─ Away: 23.0%                 │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────────┐
        │    ENSEMBLE KOMBİNASYONU           │
        ├─────────────────────────────────────┤
        │ Model Tahminleri:                  │
        │ ├─ Poisson: H:40%, D:35%, A:25%   │
        │ ├─ Dixon-Coles: H:41%, D:36%, A:23%│
        │ ├─ XGBoost: H:39%, D:34%, A:27%   │
        │ ├─ Hybrid ML: H:42%, D:34%, A:24% │
        │ ├─ CRF: H:40%, D:36%, A:24%       │
        │ ├─ Self-Learning: H:41%, D:35%, A:24%│
        │ └─ Neural Net: H:40%, D:35%, A:25%│
        │                                    │
        │ Dinamik Ağırlıklar (GA Optimize): │
        │ ├─ Poisson: 12%                   │
        │ ├─ Dixon-Coles: 18%               │
        │ ├─ XGBoost: 16%                   │
        │ ├─ Hybrid ML: 14%                 │
        │ ├─ CRF: 13%                       │
        │ ├─ Self-Learning: 15%             │
        │ └─ Neural Net: 12%                │
        │                                    │
        │ Cross-League Düzeltme:            │
        │ └─ (Farklı lig takımları için +5%)│
        │                                    │
        │ Final Weighted Average:           │
        │ ├─ Home: 40.6%                    │
        │ ├─ Draw: 34.9%                    │
        │ └─ Away: 24.5%                    │
        │                                    │
        │ Tutarlılık Kontrolü:              │
        │ ├─ En olası skor: 1-1 (18.5%)    │
        │ ├─ Skor sonucu: Draw              │
        │ └─ Tahmin uyumu: OK ✓             │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │   GÜVEN VE SON AYARLAMALAR     │
        ├─────────────────────────────────┤
        │ • Model Anlaşma Skoru: 0.89    │
        │ • Veri Kalitesi: 85%           │
        │ • Bağlam Uygunluğu: 78%        │
        │ • Final Güven: 72%              │
        │ • Uyar Seviyesi: NORMAL         │
        │                                 │
        │ Açıklamalar (XAI):             │
        │ "2 takım çok benzer güçte,     │
        │  beraberlik en olası sonuç.    │
        │  Home kadar Away'i oynatır."   │
        └─────────────────────────────────┘
```

---

## Lambda Hesaplaması - Çapraz Sistem

### 📍 Dosya: `match_prediction.py` (Line 608)

```python
lambda_home, lambda_away = self.xg_calculator.calculate_lambda_cross(
    home_xg, home_xga, away_xg, away_xga, elo_diff,
    home_team_data=home_data,
    away_team_data=away_data,
    match_context=match_context_for_lambda
)
```

### 🔢 Lambda Hesaplama Formülü

```
ADIM 1: TEMELİ λ (XG Bazlı)
────────────────────────────
λ = (xG × 0.876) + (Goals × 0.124)

Örnek:
• Home XG: 1.85
• Home Actual Goals (last 5): 7 (1.4 avg)
• λ_home_base = (1.85 × 0.876) + (1.4 × 0.124)
             = 1.622 + 0.174
             = 1.796 ≈ 1.80


ADIM 2: ELO AYARLAMASI
──────────────────────
Elo_diff = abs(Home_Elo - Away_Elo)

Ev sahibi favori ise (Elo_diff > 200):
  • +5% to +15% boost uygulanır
  
Deplasman favori ise (Elo_diff > 200, negative):
  • -5% to -15% düzeltme uygulanır

Örnek:
• Home Elo: 1850
• Away Elo: 1650
• Elo_diff: 200 (Home favori)
• Adjustment: +8%
• λ_home = 1.80 × 1.08 = 1.944 ≈ 1.95


ADIM 3: FORM BOOUTU (Son 5 Maç)
───────────────────────────────
Form_score = (Wins × 3 + Draws × 1) / Matches

Örnek:
• Home: 3 Wins, 1 Draw, 1 Loss = (9 + 1) / 5 = 2.0
• Form_boost = +10% (iyi form)
• λ_home = 1.95 × 1.10 = 2.145 ≈ 2.15

• Away: 2 Wins, 0 Draws, 3 Losses = 6 / 5 = 1.2
• Form_boost = -8% (kötü form)
• λ_away = original × 0.92


ADIM 4: CROSS-LEAGUE AYARLAMASI
────────────────────────────────
Eğer Home ve Away farklı ligdeyse:

├─ UEFA Competition (CL, EL, Super Cup):
│  └─ +20% boost (yüksek seviye)
│
├─ Aynı Seviye Ligse (her ikisi de Super Lig, vb):
│  └─ ×1.0 (ayarlama yok)
│
├─ Ev sahibi daha güçlü ligde:
│  ├─ 1 tier fark: +10% (Ev) / -10% (Deplasman)
│  └─ 2+ tier fark: +15% (Ev) / -15% (Deplasman)
│
└─ Deplasman daha güçlü ligde:
   └─ Tersine uygulanır


ADIM 5: VENUE-SPECIFIC AYARLAMASI
──────────────────────────────────
Ev sahibi (65% weight):
  • Ev maçlarında: last_5_home_avg_goals
  • Boost: +8% (ev avantajı)

Deplasman (35% weight):
  • Deplasman maçlarında: last_5_away_avg_goals
  • Düzeltme: -3% (deplasman zor)

Örnek:
• Home last 5 home games avg: 1.60 gol
• Away last 5 away games avg: 0.80 gol

Final λ'lar:
├─ λ_home = 2.15 × 1.08 = 2.322 ≈ 2.32
└─ λ_away = X × 0.97 = Y.YY


ADIM 6: KÜTÜ/YAZ EKİ (Varsa)
─────────────────────────────
Kış (Kasım-Mart): Gol sayısı ±5% değişebilir
Yaz (Haziran-Ağustos): Daha az isabetli şutlar


ADIM 7: KÜÇÜLTMESİ / BÜYÜTÜLMESI
─────────────────────────────────
λ ≥ 4.0 (ekstrem maç):
  • Ekstrem maç ayarlaması
  • 15×15 matris (normal 10×10)

λ < 0.5 (çok düşük):
  • Minimum 0.7'ye yükselt


FINAL RESULT:
────────────
λ_home: 2.32
λ_away: 1.45
```

### 🔗 Lambda Girişleri ve Bağlantıları

```
XG CALCULATOR INPUTS (calculate_lambda_cross):
├─ home_xg: float (ortalama beklenen gol)
├─ home_xga: float (defalı alınan gol)
├─ away_xg: float
├─ away_xga: float
├─ elo_diff: float (rating farkı)
├─ home_team_data: dict
│  ├─ recent_matches: list of match dicts
│  ├─ form_analysis: str (W/L/D pattern)
│  ├─ home_performance: dict
│  │  ├─ avg_goals
│  │  ├─ avg_conceded
│  │  ├─ last_5_avg_goals
│  │  └─ last_5_avg_conceded
│  └─ domestic_league_id: int (cross-league için)
│
├─ away_team_data: dict (same structure)
│
└─ match_context: dict
   ├─ league_name: str
   ├─ h2h_data: dict
   ├─ is_derby: bool
   ├─ rest_days: int
   └─ motivation_level: str

ÇIKIŞLAR:
├─ lambda_home: 2.32 (Ev sahibi beklenen gol)
└─ lambda_away: 1.45 (Deplasman beklenen gol)

BAĞLANTILAR:
lambda → POISSON MODEL
      → DIXON-COLES MODEL
      → MONTE CARLO SİMULATÖRÜ
      → MATCH CONTEXT'E YAZILIR
```

---

## Kesin Skor Tahmini

### 📍 Dosyalar ve Fonksiyonlar

```
POISSON MODEL:
├─ File: algorithms/poisson_model.py
├─ Func: calculate_probability_matrix(λ_home, λ_away, elo_diff)
└─ Out: Olasılık matrisi (11x11 array)

DIXON-COLES MODEL:
├─ File: algorithms/dixon_coles.py
├─ Func: calculate_probability_matrix(λ_home, λ_away, elo_diff)
└─ Out: Düzeltilmiş olasılık matrisi

EXACT SCORE EXTRACTION:
├─ File: algorithms/poisson_model.py
├─ Func: get_exact_score_probabilities(matrix, top_n=5)
└─ Out: List of {score: "X-Y", probability: P}
```

### 🎯 Kesin Skor Hesaplama Adımları

```
ADIM 1: POISSON MATRİSİ OLUŞTUR
────────────────────────────────
λ_home = 2.32, λ_away = 1.45

Poisson PMF: P(X=k) = (e^-λ × λ^k) / k!

P(0 goals, home) = (e^-2.32 × 2.32^0) / 0! = 0.0985
P(1 goal, home) = (e^-2.32 × 2.32^1) / 1! = 0.2286
P(2 goals, home) = (e^-2.32 × 2.32^2) / 2! = 0.2652
...

P(0 goals, away) = (e^-1.45 × 1.45^0) / 0! = 0.2347
P(1 goal, away) = (e^-1.45 × 1.45^1) / 1! = 0.3404
P(2 goals, away) = (e^-1.45 × 1.45^2) / 2! = 0.2469
...


ADIM 2: MATRİS HESAPLA
──────────────────────
Matris[h][a] = P(h goals | home) × P(a goals | away)

Örnek:
Matrix[0][0] = P(0|home) × P(0|away) = 0.0985 × 0.2347 = 0.0231 (2.31%)
Matrix[1][1] = 0.2286 × 0.3404 = 0.0778 (7.78%)
Matrix[1][0] = 0.2286 × 0.2347 = 0.0537 (5.37%)
Matrix[2][1] = 0.2652 × 0.3404 = 0.0903 (9.03%)
...

11×11 MATRIS (0-10 gol):
```
     Away: 0     1     2     3     4    5  ...
Home
  0:       2.31% 3.35% 2.43% 1.18% ...
  1:       5.37% 7.78% 5.63% 2.74% ...
  2:       6.21% 9.03% 6.55% 3.18% ...
  3:       4.79% 6.95% 5.04% 2.45% ...
  4:       2.78% 4.03% 2.92% 1.42% ...
  5:       1.29% 1.87% 1.36% 0.66% ...
 ...
```

ADIM 3: DIXON-COLES DÜZELTME (İsteğe bağlı)
─────────────────────────────────────────────
Düşük skorlara özel düzeltme (0-0, 1-0, 0-1, 1-1):

τ(0,0) = 1 - λ_home × λ_away × 0.05
       = 1 - 2.32 × 1.45 × 0.05
       = 1 - 0.168 = 0.832 (0.832 ile çarp)

τ(0,1) = 1 + λ_home × 0.05
       = 1 + 2.32 × 0.05 = 1.116

τ(1,0) = 1 + λ_away × 0.05
       = 1 + 1.45 × 0.05 = 1.073

τ(1,1) = 1 - 0.05 = 0.95


ADIM 4: NORMALIZE ET
───────────────────
Tüm matris değerleri = toplam %100 olacak şekilde bölü


ADIM 5: KESIN SKORLARI ÇIKAR (Top 5)
──────────────────────────────────────
Tüm matris hücrelerini olasılığa göre sırala:

1. Score: 1-1, Probability: 7.21%
2. Score: 2-1, Probability: 6.52%
3. Score: 1-0, Probability: 5.37%
4. Score: 2-2, Probability: 4.89%
5. Score: 0-1, Probability: 4.76%


ADIM 6: 1X2 SONUÇLARINI ÇIKAR
──────────────────────────────
Home Win (Home > Away):
  = P(1-0) + P(2-0) + P(2-1) + ... + P(10-9)
  = 5.37% + ... = 40.2%

Draw (Home = Away):
  = P(0-0) + P(1-1) + P(2-2) + ...
  = 2.31% + 7.21% + 4.89% + ... = 35.8%

Away Win (Away > Home):
  = P(0-1) + P(0-2) + P(1-2) + ... + P(9-10)
  = 4.76% + ... = 24.0%

BERABERLIK KONTROL:
├─ Minimum: 15% ✓
├─ Güncel: 35.8% ✓
└─ OK, ayarlama yok

FINAL ÇIKIŞLAR:
├─ Home Win: 40.2%
├─ Draw: 35.8%
├─ Away Win: 24.0%
├─ Over 2.5: 62.1%
├─ BTTS: 51.3%
└─ Exact Scores: [1-1: 7.21%, 2-1: 6.52%, ...]
```

### 🔗 Kesin Skor Bağlantıları

```
LAMBDA → POISSON/DIXON-COLES
  ├─ Input: (λ_home, λ_away, elo_diff)
  └─ Process: Olasılık matrisi oluştur
      │
      ├─ get_match_probabilities() → 1X2
      ├─ get_goals_probabilities() → O/U, BTTS
      └─ get_exact_score_probabilities() → Kesin skorlar
          │
          └─ EXACT SCORES (Top N)
              ├─ Kullanıldığı yerler:
              │  ├─ Correct Score Market
              │  ├─ Ensemble tutarlılık kontrolü
              │  ├─ Tahmin açıklamaları
              │  └─ HT/FT hesaplamalarında
              │
              └─ Tahmin değişkenleri:
                 ├─ final_prediction['exact_scores']: list
                 ├─ final_prediction['most_likely_score']: str
                 └─ model_predictions[model_name]['score_probabilities']: list
```

---

## Tahmin Türleri ve Bağlantıları

### 📊 Tüm Tahmin Türleri ve Kökenler

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION TYPES HIERARCHY                    │
└─────────────────────────────────────────────────────────────────┘

1. BASIC PREDICTIONS (Doğrudan λ'dan)
   ├─ Home Win: Matris[h>a] = 40.2%
   ├─ Draw: Matris[h=a] = 35.8%
   ├─ Away Win: Matris[h<a] = 24.0%
   └─ Expected Goals: λ_home, λ_away

2. GOAL-BASED PREDICTIONS (Matris analizi)
   ├─ Over 2.5: Matris[h+a>2.5] = 62.1%
   ├─ Under 2.5: Matris[h+a≤2.5] = 37.9%
   ├─ Over 1.5: 80.5%
   ├─ Under 1.5: 19.5%
   ├─ Over 3.5: 45.2%
   └─ Under 3.5: 54.8%

3. BTTS PREDICTIONS (Gol dağılımı)
   ├─ BTTS Yes: Matris[h>0 AND a>0] = 51.3%
   ├─ BTTS No: Matris[h=0 OR a=0] = 48.7%
   └─ Variants:
      ├─ BTTS Yes & Over 2.5
      └─ BTTS No & Under 2.5

4. SCORE-BASED PREDICTIONS
   ├─ Correct Score: [1-1: 7.21%, 2-1: 6.52%, ...]
   ├─ Score Lines: 0-0, 1-0, 0-1, 1-1, 2-0, ...
   └─ Goals Ranges:
      ├─ 0 Goals: 4.21%
      ├─ 1 Goal: 18.2%
      ├─ 2 Goals: 31.4%
      ├─ 3 Goals: 24.2%
      └─ 4+ Goals: 21.97%

5. HANDICAP PREDICTIONS (λ adjustment)
   ├─ Home +0.5 (deplasmana 0.5 avantaj veriyoruz):
   │  New λ_away = 1.45 + 0.5 = 1.95
   │  → Yeni 1X2: H:33%, D:33%, A:34%
   │
   ├─ Home +1.0:
   │  New λ_away = 2.45
   │  → Yeni 1X2: H:28%, D:32%, A:40%
   │
   └─ Away -1.0 (same as Home +1.0)

6. TEAM GOALS PREDICTIONS (Tekil takım analizi)
   ├─ Home Team Total:
   │  Over 1.5: Sum(Matris[2+][*]) = 55.3%
   │  Over 2.5: Sum(Matris[3+][*]) = 32.1%
   │
   └─ Away Team Total:
      Over 1.5: Sum(Matris[*][2+]) = 41.5%
      Over 2.5: Sum(Matris[*][3+]) = 15.2%

7. DOUBLE CHANCE (Bileşik sonuç)
   ├─ Home or Draw: H% + D% = 40.2% + 35.8% = 76.0%
   ├─ Away or Draw: A% + D% = 24.0% + 35.8% = 59.8%
   └─ Home or Away: H% + A% = 40.2% + 24.0% = 64.2%

8. HALF-TIME / FULL-TIME (HT/FT)
   ├─ Hesaplama:
   │  λ_ht = λ_ft / 2 (45 dakika yerine 90 dakika)
   │
   ├─ Örnek:
   │  λ_home_ft = 2.32
   │  λ_home_ht = 2.32 / 2 = 1.16
   │
   └─ Tüm Kombinasyonlar (9 sonuç):
      ├─ HT: Home/Draw/Away (3 seçenek)
      └─ FT: Home/Draw/Away (3 seçenek)
          = 3 × 3 = 9 kombinasyon

      Örnek:
      H/H (Home HT, Home FT): 12.3%
      H/D (Home HT, Draw FT): 8.1%
      H/A (Home HT, Away FT): 4.2%
      D/H (Draw HT, Home FT): 11.5%
      D/D (Draw HT, Draw FT): 8.9%
      ... ve devamı

9. GOAL RANGE (Gol sayısı aralıkları)
   ├─ 0-1 Goals: Matris[<2] = 19.5%
   ├─ 1-2 Goals: (1-2 goals) = 30.2%
   ├─ 2-3 Goals: (2-3 goals) = 31.4%
   ├─ 3+ Goals: Matris[3+] = 24.2%
   └─ Exact Goals: (0, 1, 2, 3, 4, ...)

10. ADVANCED PREDICTIONS (ML Modelleri)
    ├─ Feature-Based (XGBoost, Neural Net)
    │  └─ Lambda ayarlaması: ±5-15%
    │
    ├─ CRF (Conditional Random Field)
    │  └─ Maç dizilimi patternleri
    │
    └─ Self-Learning
       └─ Geçmiş tahmin hataları dikkate alır

11. PSYCHOLOGICAL ADJUSTMENTS
    ├─ Motivasyon Bonus: ±5-10%
    ├─ Momentum Effect: ±3-8%
    ├─ Derbi Faktor: ±5-12%
    └─ Draw Floor: Minimum 15%

12. FINAL ENSEMBLE
    ├─ Tüm tahminlerin ağırlıklı ortalaması
    ├─ Dinamik ağırlıklar (GA optimized)
    ├─ Cross-league adjustment
    ├─ Meta-learning akıllı seçim
    └─ Confidence scoring (45-90%)
```

---

## Ensemble Kombinasyon Sistemi

### 📍 Dosya: `algorithms/ensemble.py`

### 🔄 Ensemble Akışı (combine_predictions)

```python
def combine_predictions(self, model_predictions, match_context, algorithm_weights):
    """
    Tüm model tahminlerini birleştir
    
    INPUT:
    ├─ model_predictions: dict
    │  ├─ 'poisson': {home_win: %, draw: %, away_win: %, ...}
    │  ├─ 'dixon_coles': {...}
    │  ├─ 'xgboost': {...}
    │  ├─ 'hybrid_ml': {...}
    │  ├─ 'crf': {...}
    │  ├─ 'self_learning': {...}
    │  └─ 'neural_network': {...}
    │
    ├─ match_context: dict
    │  ├─ lambda_home, lambda_away
    │  ├─ elo_diff
    │  ├─ cross_league: bool
    │  ├─ league_strength_context: dict
    │  └─ ...
    │
    └─ algorithm_weights: dict
       ├─ 'poisson': 12%
       ├─ 'dixon_coles': 18%
       ├─ ... vb
       └─ (genetik algoritma optimized)
    """
    
    # 1. Ağırlık ayarlaması (Bağlama göre)
    adjusted_weights = self._adjust_weights_by_context(weights, context)
    
    # 2. Meta-Learning Layer (Akıllı model seçimi)
    if self.use_meta_learning:
        model_selection = self.meta_learning_layer.select_best_models(
            model_predictions, match_context
        )
        # Başarılı modellere daha yüksek ağırlık
        adjusted_weights = apply_meta_learning_weights(model_selection)
    
    # 3. Temel Ensemble (Ağırlıklı ortalama)
    combined = {
        'home_win': sum(predictions['home_win'] * weight for ...),
        'draw': sum(predictions['draw'] * weight for ...),
        'away_win': sum(predictions['away_win'] * weight for ...),
        # ... diğer pazarlar
    }
    
    # 4. Tutarlılık Kontrolü (En olası skor ile)
    if max_score_prob > 3%:
        if score_outcome != combined_outcome and diff < 10%:
            # En olası skora göre ayarla (+8%)
            combined['draw'] += adjustment  # veya home/away
    
    # 5. Beraberlik Minimum Sınırı
    if combined['draw'] < 15%:
        combined['draw'] = 15%
        # Eksik miktarı ev/deplasmandan çıkar
        # (Her sonuç min 5% kalmalı)
    
    # 6. Tek Sonuç Maksimum Sınırı
    if combined['home_win'] > 75%:
        excess = combined['home_win'] - 75%
        combined['home_win'] = 75%
        combined['draw'] += excess × 0.6
        combined['away_win'] += excess × 0.4
    
    # 7. Normalize Et (Toplam 100%)
    total = combined['home_win'] + combined['draw'] + combined['away_win']
    combined['home_win'] = (combined['home_win'] / total) × 100
    # ... others
    
    # 8. Cross-League Adjustment (Farklı lig takımları)
    if match_context['cross_league']:
        league_context = match_context['league_strength_context']
        
        if is_uefa_competition:
            # UEFA ligi: +20% Home, -10% Away
            home_boost = 1.20
            away_factor = 0.85
        elif home_stronger:
            # Home daha güçlü ligde
            home_boost = 1.10
            away_factor = 0.90
        else:
            # Away daha güçlü ligde
            home_boost = 0.85
            away_factor = 1.15
        
        combined['home_win'] *= home_boost
        combined['away_win'] *= away_factor
        # Normalize
    
    # 9. Güven Hesaplaması (Advanced Confidence System)
    confidence = calculate_comprehensive_confidence(
        model_predictions, match_context, combined
    )
    combined['confidence'] = confidence  # 45-90%
    
    # 10. Uyar Seviyeleri
    if model_agreement < 0.7:
        combined['alert_level'] = 'HIGH'  # Düşük uyum
    elif confidence < 55:
        combined['alert_level'] = 'MEDIUM'  # Düşük güven
    else:
        combined['alert_level'] = 'NORMAL'
    
    return combined
```

### ⚖️ Dinamik Ağırlık Sistemi

```
BAŞLANGIÇ AĞIRLIKLARI (Varsayılan):
├─ Poisson: 12%
├─ Dixon-Coles: 18% (düşük skorlarda iyi)
├─ XGBoost: 16% (feature-based)
├─ Hybrid ML: 14% (ELO+Form kombinesi)
├─ CRF: 13% (sıra patternleri)
├─ Self-Learning: 15% (geçmiş hataları)
└─ Neural Network: 12% (kompleks patterns)

GAG (Genetic Algorithm) OPTİMİZASYONU:
├─ Populasyon: 30 ağırlık seti
├─ Dönem: 50 iterasyon
├─ Elite: En iyi 6 seti koru
├─ Mutasyon: %20 rastgele değişim
└─ Sonuç: Geçmiş maçlara en uygun ağırlıklar

BAĞLAMA GÖRE AYARLAMA:
├─ Ev Maçı Avantajı:
│  ├─ Dixon-Coles ↑ (+2%)
│  ├─ Hybrid ML ↑ (+1%)
│  └─ Self-Learning ↑ (+1%)
│
├─ Deplasman Zor:
│  ├─ Poisson ↓ (-1%)
│  └─ Neural Net ↑ (+1%)
│
├─ İyi Form (W/W/W):
│  ├─ Self-Learning ↑ (+3%)
│  └─ Neural Net ↑ (+2%)
│
├─ Kötü Form (L/L/L):
│  ├─ Self-Learning ↓ (-2%)
│  └─ CRF ↑ (+2%)
│
├─ Yüksek ELO Farkı (>300):
│  ├─ Poisson ↑ (+2%)
│  └─ Self-Learning ↓ (-1%)
│
└─ Dengeli Takımlar (<50 Elo farkı):
   ├─ CRF ↑ (+3%)
   └─ Hybrid ML ↑ (+2%)
```

### 🧠 Meta-Learning Layer

```
Meta-Learning Layer, hangi modellerin hangi durumlarda
en iyi tahmin yaptığını öğrenir.

ÖRNEK ÖĞRENILMIŞ PATTERN:

"Ev sahibi favori (λ_home > 2.0 ve Elo_diff > 200):
  ├─ Dixon-Coles: %87 doğruluk
  ├─ Poisson: %84 doğruluk
  ├─ Self-Learning: %79 doğruluk
  └─ → Dixon-Coles ve Poisson'u ön plana çık"

"Dengeli maç (λ yakın ve Elo < 50 farkı):
  ├─ CRF: %76 doğruluk (sıra patternleri)
  ├─ Neural Net: %74 doğruluk
  └─ → CRF'ye daha yüksek ağırlık ver"

"Kötü form takım dönmek üzere (3L sonra W):
  ├─ Self-Learning: %81 doğruluk
  ├─ Hybrid ML: %77 doğruluk
  └─ → Self-Learning etkin"

ÇIKIŞLAR:
├─ selected_models: ['dixon_coles', 'poisson', 'self_learning']
├─ confidence_multipliers: {model: 0.9-1.1}
└─ reason_explanation: str
```

---

## Implementation Detayları

### 🔧 Asıl Kod Referansları

#### Lambda Hesaplaması
```python
# File: algorithms/xg_calculator.py
def calculate_lambda_cross(self, home_xg, home_xga, away_xg, away_xga, elo_diff,
                          home_team_data, away_team_data, match_context):
    """
    Çapraz lambda hesapla
    
    xG bazında hesaplama:
    lambda_home = (home_xg × 0.876) + (avg_goals × 0.124)
    
    Ayarlamalar:
    1. ELO farkına göre: ±5-15%
    2. Form boost: last 5 matches
    3. Cross-league: ±10-20%
    4. Venue bonus: ±8%
    5. Rest days: ±2-5%
    """
```

#### Poisson Model
```python
# File: algorithms/poisson_model.py
def calculate_probability_matrix(self, lambda_home, lambda_away, elo_diff=0):
    """
    Poisson olasılık matrisi oluştur
    
    Favori boost (1.15x) beraberlik koruma ile:
    1. Orijinal beraberlik toplamını kaydet
    2. Favori bonusu uygula (yüksek skorlara)
    3. Beraberlik kayıplarını geri ekle
    4. Normalize et
    """
```

#### Kesin Skor Çıkarımı
```python
# File: algorithms/poisson_model.py
def get_exact_score_probabilities(self, prob_matrix, top_n=5):
    """
    Olasılık matrisinden kesin skorları çıkar
    
    1. Tüm matris hücrelerini yaz (h, a, prob)
    2. Olasılığa göre sırala
    3. En yüksek N'yi döndür
    4. Yüzde formatında
    """
```

#### Ensemble Kombinasyon
```python
# File: algorithms/ensemble.py
def combine_predictions(self, model_predictions, match_context, algorithm_weights):
    """
    Tüm modelleri birleştir
    
    1. Ağırlık ayarla (GA optimized)
    2. Meta-learning: akıllı model seçimi
    3. Temel ensemble (ağırlıklı ortalama)
    4. Tutarlılık kontrolü
    5. Beraberlik minimum sınırı (%15)
    6. Tek sonuç maksimum sınırı (%75)
    7. Normalize et (%100)
    8. Cross-league adjustment
    9. Güven hesapla
    10. Final tahmin
    """
```

### 📊 Çıkış Formatı (JSON)

```json
{
  "match_id": 12345,
  "home_team": "Galatasaray",
  "away_team": "Fenerbahçe",
  
  "primary_predictions": {
    "1x2": {
      "home_win": 40.6,
      "draw": 34.9,
      "away_win": 24.5
    },
    "exact_scores": [
      {"score": "1-1", "probability": 7.21},
      {"score": "2-1", "probability": 6.52},
      {"score": "1-0", "probability": 5.37},
      {"score": "2-2", "probability": 4.89},
      {"score": "0-1", "probability": 4.76}
    ],
    "over_under": {
      "over_2_5": 62.1,
      "under_2_5": 37.9
    },
    "btts": {
      "yes": 51.3,
      "no": 48.7
    }
  },
  
  "advanced_predictions": {
    "halftime_fulltime": {
      "h_h": 12.3,
      "h_d": 8.1,
      "h_a": 4.2,
      "d_h": 11.5,
      "d_d": 8.9,
      "d_a": 5.3,
      "a_h": 6.2,
      "a_d": 4.8,
      "a_a": 6.0
    },
    "handicap": {
      "home_plus_0_5": {"1": 43.2, "X": 32.1, "2": 24.7},
      "home_plus_1_0": {"1": 38.5, "X": 31.8, "2": 29.7}
    },
    "team_goals": {
      "home_over_1_5": 55.3,
      "home_over_2_5": 32.1,
      "away_over_1_5": 41.5,
      "away_over_2_5": 15.2
    }
  },
  
  "lambda_values": {
    "home": 2.32,
    "away": 1.45,
    "expected_total": 3.77
  },
  
  "model_predictions": {
    "poisson": {...},
    "dixon_coles": {...},
    "xgboost": {...},
    "hybrid_ml": {...},
    "crf": {...},
    "self_learning": {...},
    "neural_network": {...}
  },
  
  "ensemble_info": {
    "weights": {
      "poisson": 0.12,
      "dixon_coles": 0.18,
      ...
    },
    "model_agreement": 0.89,
    "prediction_variance": 0.15,
    "meta_learning_applied": true,
    "cross_league_adjustment": false
  },
  
  "confidence": {
    "overall": 72,
    "model_agreement": 0.89,
    "data_quality": 85,
    "context_familiarity": 78,
    "stability_score": 81,
    "alert_level": "NORMAL",
    "recommendation_strength": "MODERATE"
  },
  
  "explanation": {
    "summary": "İki takım benzer güçte...",
    "key_factors": [
      "Ev sahibi form: Excellent (3W-1D)",
      "Deplasman zor: Average form (2W-3L)",
      "ELO fark: 200 (Home avantaj)",
      "Lig: Aynı lig, ayarlama yok"
    ],
    "prediction_reasoning": "..."
  }
}
```

---

## 🔍 Klonlama Rehberi

Bu sistemi başka bir yerden klonlamak için gerekli dosyalar:

### Temel Dosyalar
```
1. match_prediction.py (Main predictor class)
2. algorithms/xg_calculator.py (Lambda calculation)
3. algorithms/poisson_model.py (Poisson distribution)
4. algorithms/dixon_coles.py (Low-score adjustment)
5. algorithms/ensemble.py (Model combination)
6. algorithms/elo_system.py (Rating system)
```

### Destek Dosyaları
```
7. algorithms/feature_extraction_pipeline.py
8. algorithms/dynamic_team_analyzer.py
9. algorithms/league_strength_analyzer.py
10. algorithms/psychological_profiler.py
```

### Veri Dosyaları
```
11. config/league_ids.json (Lig ID mappings)
12. config/league_strength.json (Lig seviyeleri)
13. football_api_config.py (API ayarları)
```

### Uygulama Akışı
1. **Başlat**: `MatchPredictor()` sınıfını oluştur
2. **Lambda**: `calculate_lambda_cross()` çağır
3. **Matris**: `calculate_probability_matrix()` çağır
4. **Skorlar**: `get_exact_score_probabilities()` çağır
5. **Ensemble**: `combine_predictions()` çağır
6. **Çıkış**: JSON formatında tahmin döndür

---

**Dokümantasyon Sürümü**: 1.0 (Aralık 2025)
**Son Güncelleme**: Beraberlik koruma sistemi, cross-lambda, ensemble optimizasyon
