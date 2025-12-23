# Çapraz Lambda (Cross-Lambda) Hesaplama Formülü

## 📐 Sistem Özeti
Sisteminizde kullanılan lambda hesaplama **7 adımlı bir kompozit sistem** ile yapılır.

---

## ADIM 1: VENUE-SPECIFIC PERFORMANS AYARLAMASI

```
AMAÇ: Son 5 ev/deplasman maçında kaç gol attığını ve yenildiğini dikkate almak

Ev Sahibi İçin:
────────────────
adjusted_home_xg = (venue_home_xg × 0.65) + (home_xg × 0.35)
adjusted_home_xga = (venue_home_xga × 0.65) + (home_xga × 0.35)

Deplasman İçin:
────────────────
adjusted_away_xg = (venue_away_xg × 0.65) + (away_xg × 0.35)
adjusted_away_xga = (venue_away_xga × 0.65) + (away_xga × 0.35)

AĞIRLIKLARIN ANLAMI:
├─ venue_xg/xga: Son 5 ev/deplasman maçından hesaplanan ortalama
│  └─ 65% → Son 5 maçın daha etkili (güncel form)
│
└─ genel xg/xga: Tüm son maçlardan hesaplanan ortalama
   └─ 35% → Daha geniş perspektif (sabitlik)


ÖRNEK:
──────
Home Team: Galatasaray vs Fenerbahçe (Galatasaray'ın evi)

Veriler:
├─ Galatasaray genel xG: 1.85 (son tüm maçlar)
├─ Galatasaray son 5 ev xG: 1.95 (son 5 ev maçı)
├─ Galatasaray genel xGA: 1.30
└─ Galatasaray son 5 ev xGA: 1.15 (daha az gol yedi evde)

Hesaplama:
├─ adjusted_home_xg = (1.95 × 0.65) + (1.85 × 0.35)
│                   = 1.268 + 0.648
│                   = 1.916 ≈ 1.92
│
└─ adjusted_home_xga = (1.15 × 0.65) + (1.30 × 0.35)
                     = 0.748 + 0.455
                     = 1.203 ≈ 1.20

Fenerbahçe (deplasman)
├─ Fenerbahçe genel xG: 1.72
├─ Fenerbahçe son 5 deplasman xG: 1.58 (deplasmanda daha zayıf)
├─ Fenerbahçe genel xGA: 1.40
└─ Fenerbahçe son 5 deplasman xGA: 1.55 (deplasmanda daha çok gol yedi)

Hesaplama:
├─ adjusted_away_xg = (1.58 × 0.65) + (1.72 × 0.35)
│                   = 1.027 + 0.602
│                   = 1.629 ≈ 1.63
│
└─ adjusted_away_xga = (1.55 × 0.65) + (1.40 × 0.35)
                     = 1.008 + 0.490
                     = 1.498 ≈ 1.50
```

---

## ADIM 2: FAVORİ TAKIMI AYARLAMASI (ELO BAZLI)

```
AMAÇ: ELO farkı çok büyükse daha dengeli bir tahmin yapmak

KOŞULı AYARLAMALAR:
───────────────────

Koşul 1: Ev favoriyse ama xG'sı az
IF: elo_diff > 0 AND home_xg < away_xg
THEN: home_xg = min(home_xg + 0.3, away_xg × 1.2)
      └─ Ev sahibini en fazla %20 yukarı çek


Koşul 2: Deplasman favoriyse ama xG'sı az
IF: elo_diff < 0 AND away_xg < home_xg
THEN: away_xg = min(away_xg + 0.3, home_xg × 1.2)
      └─ Deplasmancıyı en fazla %20 yukarı çek


Koşul 3: Ev savunması çok zayıfsa
IF: elo_diff > 0 AND home_xga > away_xga × 1.2
THEN: home_xga = max(home_xga - 0.3, away_xga × 0.8)
      └─ Ev savunmasını en fazla %20 aşağı çek


Koşul 4: Deplasman savunması çok zayıfsa
IF: elo_diff < 0 AND away_xga > home_xga × 1.2
THEN: away_xga = max(away_xga - 0.3, home_xga × 0.8)
      └─ Deplasman savunmasını en fazla %20 aşağı çek


ÖRNEK:
──────
Elo_diff = +250 (Galatasaray çok favori)
home_xg = 1.60 (ama Fenerbahçe xG'dan daha az)
away_xg = 1.85

Uygulama:
├─ Koşul kontrol: elo_diff > 0 ✓ AND home_xg < away_xg ✓
├─ Yeni home_xg = min(1.60 + 0.3, 1.85 × 1.2)
│               = min(1.90, 2.22)
│               = 1.90
│
└─ Sonuç: Galatasaray xG'sı 1.60 → 1.90 (favoriliğini yansıtmak için +0.30)
```

---

## ADIM 3: VENUE BONUS (SON 5 EV/DEPLASMAN MAÇI)

```
AMAÇ: Son 5 ev/deplasman maçında başarılıysa bonus vermek

Ev Sahibi İçin:
────────────────
IF: home_performance['last_5_win_rate'] > 0.60 (3+ win in 5)
    THEN: home_venue_bonus = 1.10 (%10 bonus)
ELSE:
    home_venue_bonus = 1.0 (ayarlama yok)


Deplasman İçin:
────────────────
IF: away_performance['last_5_win_rate'] > 0.40 (2+ win in 5)
    THEN: away_venue_bonus = 1.05 (%5 bonus - deplasmada kazanmak daha zor)
ELSE:
    away_venue_bonus = 1.0 (ayarlama yok)


ÖRNEK:
──────
Galatasaray son 5 ev maçı: W-W-W-L-D = 3 kazanış / 5 = 0.60 win rate
├─ Kontrol: 0.60 > 0.60? HAYIR (tam eşit)
├─ Koşul: > 0.60 (strict greater than)
└─ home_venue_bonus = 1.0 (bonus verilmez)

Ancak: 3 kazanış / 5 olsaydı = 0.61
├─ Kontrol: 0.61 > 0.60? EVET
└─ home_venue_bonus = 1.10 (%10 bonus)


Fenerbahçe son 5 deplasman maçı: W-D-D-L-L = 1 kazanış / 5 = 0.20 win rate
├─ Kontrol: 0.20 > 0.40? HAYIR
└─ away_venue_bonus = 1.0 (bonus yok)
```

---

## ADIM 4: LİG FAKTÖRÜ (SADECE FARKLI LİGLER İÇİN!)

```
AMAÇ: Farklı liglerden takımlar arası gücü dengelemek

KURAL:
──────
IF: home_league ≠ away_league (farklı ligde)
THEN: 
    home_league_factor = LeagueAnalyzer.analyze(home_league)['lambda_factor']
    away_league_factor = LeagueAnalyzer.analyze(away_league)['lambda_factor']
    league_factor = (home_league_factor + away_league_factor) / 2
ELSE IF: is_cup_match
THEN:
    league_factor = 1.05 (kupa maçları daha dinamik)
ELSE:
    league_factor = 1.0 (aynı lig - nötr)


LİG FAKTÖRLERI ÖRNEKLERI:
──────────────────────────
├─ Champions League (UEFA): 1.20
├─ Europa League (UEFA): 1.15
├─ Super Lig (Türkiye): 1.00 (referans)
├─ Premier League (İngiltere): 1.10 (daha yüksek gol)
├─ Serie A (İtalya): 0.95 (daha düşük gol)
├─ Bundesliga (Almanya): 1.15 (saldırgan)
├─ Ligue 1 (Fransa): 1.05
├─ Dünya Kupası: 1.25
└─ Lig Kupası: 1.08


ÖRNEK:
──────
Galatasaray (Super Lig, λ_factor=1.00) vs Liverpool (Premier League, λ_factor=1.10)

UEFA Şampiyonlar Ligi maçı:
├─ home_league_factor = 1.00 (Super Lig)
├─ away_league_factor = 1.10 (Premier League)
├─ league_factor = (1.00 + 1.10) / 2 = 1.05

Sonuç: %5 daha fazla gol beklenir (UEFA ligi vs farklı ligler)
```

---

## ADIM 5: AĞIRLIKLI ORTALAMA FAKTÖRÜ HESAPLAMA

```
AMAÇ: Tüm düzeltmeleri (log, venue, league) dengeli bir şekilde birleştirmek

A. TEMELİ LAMBDA HESAPLA:
──────────────────────────
base_lambda_home = home_xg × away_xga
base_lambda_away = away_xg × home_xga

Bu, "ev sahibi atak gücü × deplasman savunma zayıflığı" anlamına gelir.


ÖRNEK (Devam):
──────────────
Yeni değerler (Adım 1-2 sonrası):
├─ home_xg = 1.92
├─ home_xga = 1.20
├─ away_xg = 1.63
├─ away_xga = 1.50

base_lambda_home = 1.92 × 1.50 = 2.880
base_lambda_away = 1.63 × 1.20 = 1.956


B. LOG DÜZELTME FAKTÖRÜ:
────────────────────────
strength_ratio = home_xg / away_xg
              = 1.92 / 1.63
              = 1.178

log_adjustment_home = 1 + 0.1 × log(strength_ratio + 1)
                    = 1 + 0.1 × log(2.178)
                    = 1 + 0.1 × 0.778
                    = 1.0778 ≈ 1.078

log_adjustment_away = 1 - 0.1 × log(strength_ratio + 1)
                    = 1 - 0.1 × 0.778
                    = 0.9222 ≈ 0.922

(Log, gücü orantılı bir şekilde ayarlar - güçlü takım daha fazla bonus, zayıf daha fazla ceza)


C. FAKTÖRLERIN AĞIRLIKLARI:
────────────────────────────
weight_log = 0.40      (%40 - log düzeltmesi en önemli)
weight_venue = 0.30    (%30 - ev/deplasman performansı)
weight_league = 0.30   (%30 - lig gücü)
                ────
TOPLAM = 1.00


D. KOMBİNE FAKTÖR (Ev Sahibi):
─────────────────────────────
combined_factor_home = (
    (weight_log × log_adjustment_home) +
    (weight_venue × home_venue_bonus) +
    (weight_league × league_factor)
) / (weight_log + weight_venue + weight_league)

= (
    (0.40 × 1.078) +
    (0.30 × 1.0) +
    (0.30 × 1.05)
) / 1.0

= (
    0.4312 +
    0.3000 +
    0.3150
) / 1.0

= 1.0462 ≈ 1.046


E. KOMBİNE FAKTÖR (Deplasman):
────────────────────────────
combined_factor_away = (
    (weight_log × log_adjustment_away) +
    (weight_venue × away_venue_bonus) +
    (weight_league × league_factor)
) / 1.0

= (
    (0.40 × 0.922) +
    (0.30 × 1.0) +
    (0.30 × 1.05)
) / 1.0

= (
    0.3688 +
    0.3000 +
    0.3150
) / 1.0

= 0.9838 ≈ 0.984
```

---

## ADIM 6: FINAL LAMBDA HESAPLAMA

```
FORMÜL:
───────
λ_home = base_lambda_home × combined_factor_home
λ_away = base_lambda_away × combined_factor_away


ÖRNEK:
──────
λ_home = 2.880 × 1.046 = 3.012 ≈ 3.01
λ_away = 1.956 × 0.984 = 1.925 ≈ 1.93

Bu λ değerleri Poisson/Dixon-Coles modeline verilir!
```

---

## ADIM 7: EKSTREM MAÇI KONTROL VE SINIRLAMA

```
AMAÇ: Çok yüksek lambda değerlerini makul sınırlar içine almak

EKSTREM MAÇI BELIRLEME:
──────────────────────
Maç ekstrem sayılırsa (örneğin λ_home + λ_away > 5.0):
└─ 15×15 olasılık matrisi kullan (normal 10×10)

SINIRLAMA KAPLARI:
──────────────────
lambda_home = max(0.5, min(lambda_cap, lambda_home))
lambda_away = max(0.5, min(lambda_cap, lambda_away))

Minimum sınır: 0.5 (çok düşük lambdalar da mümkün değil)
Maksimum sınır: Ekstrem maç tespit algoritması tarafından belirlenilir
                (genelde 4.0-4.5 arası)


ÖRNEK:
──────
λ_home = 3.01 (normal aralık) → Sınırlandırma yok ✓
λ_away = 1.93 (normal aralık) → Sınırlandırma yok ✓

Toplam = 3.01 + 1.93 = 4.94 → Ekstrem maç mi? (>5.0? HAYIR)
└─ 10×10 Poisson matrisi kullan

(Ancak λ_home + λ_away > 5.0 olsaydı → ekstrem maç, 15×15 matris)
```

---

## 📊 COMPLETE FORMULA TREE

```
┌────────────────────────────────────────────────────────────┐
│                    CROSS-LAMBDA FORMULA                     │
└────────────────────────────────────────────────────────────┘

INPUT VALUES:
├─ home_xg, home_xga (takım/rakip xG)
├─ away_xg, away_xga
├─ elo_diff (ELO farkı)
└─ match_context (lig, venue, derbi, vb)

ADIM 1: Venue-Specific Adjustment (65% weight)
├─ adjusted_home_xg = (venue_home_xg × 0.65) + (home_xg × 0.35)
├─ adjusted_home_xga = (venue_home_xga × 0.65) + (home_xga × 0.35)
├─ adjusted_away_xg = (venue_away_xg × 0.65) + (away_xg × 0.35)
└─ adjusted_away_xga = (venue_away_xga × 0.65) + (away_xga × 0.35)

ADIM 2: Favorite Team Adjustment (ELO-based)
├─ IF elo_diff > 0:
│  ├─ IF home_xg < away_xg → home_xg += 0.3 (max 1.2× away)
│  └─ IF home_xga > away_xga×1.2 → home_xga -= 0.3 (min 0.8× away)
└─ (Reverse for elo_diff < 0)

ADIM 3: Venue Bonus
├─ IF home_last_5_win_rate > 0.60 → home_venue_bonus = 1.10
├─ ELSE → home_venue_bonus = 1.0
├─ IF away_last_5_win_rate > 0.40 → away_venue_bonus = 1.05
└─ ELSE → away_venue_bonus = 1.0

ADIM 4: League Factor
├─ IF home_league ≠ away_league:
│  └─ league_factor = (home_league_factor + away_league_factor) / 2
├─ ELSE IF cup_match:
│  └─ league_factor = 1.05
└─ ELSE:
   └─ league_factor = 1.0

ADIM 5: Base Lambda
├─ base_lambda_home = adjusted_home_xg × adjusted_away_xga
└─ base_lambda_away = adjusted_away_xg × adjusted_home_xga

ADIM 6: Combined Factor
├─ strength_ratio = adjusted_home_xg / adjusted_away_xg
│
├─ log_adj_home = 1 + 0.1 × log(strength_ratio + 1)
├─ log_adj_away = 1 - 0.1 × log(strength_ratio + 1)
│
├─ combined_factor_home = 
│   (0.40 × log_adj_home + 0.30 × home_venue_bonus + 0.30 × league_factor)
│
└─ combined_factor_away = 
    (0.40 × log_adj_away + 0.30 × away_venue_bonus + 0.30 × league_factor)

ADIM 7: Final Lambda
├─ λ_home = base_lambda_home × combined_factor_home
└─ λ_away = base_lambda_away × combined_factor_away

ADIM 8: Extreme Match Check & Capping
├─ IF (λ_home + λ_away) > 5.0 → Extreme match detected
├─ lambda_home = max(0.5, min(lambda_cap, lambda_home))
└─ lambda_away = max(0.5, min(lambda_cap, lambda_away))

OUTPUT:
└─ (λ_home, λ_away) → Poisson/Dixon-Coles modeline gider
```

---

## 🔢 COMPLETE WORKED EXAMPLE

```
MAÇLAR: Galatasaray (Home) vs Fenerbahçe (Away)

INPUT VERILERI:
───────────────
HOME:
├─ home_xg = 1.85
├─ home_xga = 1.30
├─ home_venue_xg (last 5) = 1.95
├─ home_venue_xga (last 5) = 1.15
├─ home_last_5_win_rate = 0.60 (W-W-W-L-D = 3/5)
└─ home_league = "Super Lig"

AWAY:
├─ away_xg = 1.72
├─ away_xga = 1.40
├─ away_venue_xg (last 5) = 1.58
├─ away_venue_xga (last 5) = 1.55
├─ away_last_5_win_rate = 0.20 (W-D-D-L-L = 1/5)
└─ away_league = "Super Lig"

OTHER:
├─ elo_diff = +150 (Gala favori)
├─ league_factor = 1.0 (aynı lig)
└─ is_cup = false


STEP 1: VENUE-SPECIFIC ADJUSTMENT
──────────────────────────────────
home_xg = (1.95 × 0.65) + (1.85 × 0.35) = 1.268 + 0.648 = 1.916
home_xga = (1.15 × 0.65) + (1.30 × 0.35) = 0.748 + 0.455 = 1.203

away_xg = (1.58 × 0.65) + (1.72 × 0.35) = 1.027 + 0.602 = 1.629
away_xga = (1.55 × 0.65) + (1.40 × 0.35) = 1.008 + 0.490 = 1.498


STEP 2: FAVORITE ADJUSTMENT
────────────────────────────
elo_diff = +150 (home favori)

Check 1: home_xg (1.916) < away_xg (1.629)? HAYIR
         → No adjustment needed

Check 2: home_xga (1.203) > away_xga (1.498) × 1.2 = 1.798? HAYIR
         → No adjustment needed

(No changes - ELO farkı önemli değil çünkü değerler mantıklı)


STEP 3: VENUE BONUS
────────────────────
home_last_5_win_rate = 0.60 > 0.60? HAYIR (tam eşit, strict >)
→ home_venue_bonus = 1.0

away_last_5_win_rate = 0.20 > 0.40? HAYIR
→ away_venue_bonus = 1.0


STEP 4: LEAGUE FACTOR
──────────────────────
home_league == away_league? EVET
→ league_factor = 1.0 (aynı lig, nötr)


STEP 5: BASE LAMBDA
────────────────────
base_lambda_home = 1.916 × 1.498 = 2.870
base_lambda_away = 1.629 × 1.203 = 1.960


STEP 6: COMBINED FACTOR
────────────────────────
strength_ratio = 1.916 / 1.629 = 1.176
log(strength_ratio + 1) = log(2.176) = 0.777

log_adj_home = 1 + 0.1 × 0.777 = 1.0777
log_adj_away = 1 - 0.1 × 0.777 = 0.9223

combined_factor_home = (0.40 × 1.0777 + 0.30 × 1.0 + 0.30 × 1.0) / 1.0
                     = (0.4311 + 0.3000 + 0.3000) / 1.0
                     = 1.0311

combined_factor_away = (0.40 × 0.9223 + 0.30 × 1.0 + 0.30 × 1.0) / 1.0
                     = (0.3689 + 0.3000 + 0.3000) / 1.0
                     = 0.9689


STEP 7: FINAL LAMBDA
──────────────────────
λ_home = 2.870 × 1.0311 = 2.959 ≈ 2.96
λ_away = 1.960 × 0.9689 = 1.899 ≈ 1.90


STEP 8: EXTREME CHECK
───────────────────────
Total λ = 2.96 + 1.90 = 4.86 < 5.0 → Normal maç (ekstrem değil)

Final check:
├─ λ_home = max(0.5, min(4.0, 2.96)) = 2.96 ✓
└─ λ_away = max(0.5, min(4.0, 1.90)) = 1.90 ✓


FINAL RESULT:
═════════════
λ_home = 2.96 (Galatasaray, ev sahibi)
λ_away = 1.90 (Fenerbahçe, deplasman)

Bu değerler Poisson modeline verilir:
├─ P(0|home) = e^-2.96 × 2.96^0 / 0! = 0.0516
├─ P(1|home) = e^-2.96 × 2.96^1 / 1! = 0.1527
├─ P(2|home) = e^-2.96 × 2.96^2 / 2! = 0.2262
│ ...
├─ P(0|away) = e^-1.90 × 1.90^0 / 0! = 0.1496
├─ P(1|away) = e^-1.90 × 1.90^1 / 1! = 0.2842
├─ P(2|away) = e^-1.90 × 1.90^2 / 2! = 0.2699
│ ...

Matristen 1X2 ve kesin skorları hesapla!
```

---

## 📝 ÖZEL DURUMLAR

### UEFA Şampiyonlar Ligi Maçı
```
İf: Home = Super Lig takımı, Away = Premier League takımı

league_factor = (Super_Lig_factor + Premier_factor) / 2
             = (1.00 + 1.10) / 2
             = 1.05

Bu %5 daha fazla gol tahmin edilmesi demek.
```

### Kış Dönemi
```
Kış aylarında gol sayısı ±5% değişebilir
(Bu, lig faktörü veya diğer ayarlama yapılmaz, 
 sadece başlangıç xG hesaplamasında dikkate alınır)
```

### Çok Zayıf Takım vs Çok Güçlü Takım
```
ELO farkı > 500 ise:
├─ Zayıf takımın λ artırılır (%5-10)
├─ Güçlü takımın λ azaltılır (%5-10)
└─ Sonuç daha dengeli bir tahmin
```

---

## 🎯 ÖZETLESİ

```
┌─────────────────────────────────────────────────────────┐
│     7-STEP CROSS-LAMBDA CALCULATION SUMMARY             │
└─────────────────────────────────────────────────────────┘

1. VENUE-SPECIFIC: Son 5 ev/deplasman maçı (65% ağırlık)
2. FAVORITE ADJ: ELO tabanlı dengeli düzeltme
3. VENUE BONUS: %10 ev, %5 deplasman (form iyi ise)
4. LEAGUE FACTOR: Farklı ligler için (+5% to ±20%)
5. BASE LAMBDA: xG × rakip xGA (core calculation)
6. COMBINED: Log(40%) + Venue(30%) + League(30%)
7. FINAL: Base × Combined + Ekstrem kontrol

AĞIRLIKLARIN YERLEŞİMİ:
├─ Log düzeltme: %40 (güç oranı)
├─ Venue bonus: %30 (son form)
└─ Lig faktörü: %30 (lig seviyesi)

ÇIKIŞLAR:
└─ λ_home, λ_away → Poisson/Dixon-Coles → 1X2 & Kesin skorlar
```

Eğer başka bir noktada detay istersen yazabilirsin! 🎯

---

**Versiyon**: 1.0
**Tarih**: Aralık 2025
**Kaynak**: algorithms/xg_calculator.py - calculate_lambda_cross()
