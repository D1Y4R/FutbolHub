// Takım İstatistikleri Popup Fonksiyonları

// API parametreleri - son 5 maç performansı için kullanılan endpoint'ler
const TEAM_STATS_API = {
    prediction: '/api/predict-match/{teamId}/{teamId}', // Query: home_name, away_name
    stats: '/api/v3/team-stats/{teamId}',
    fixtures: '/api/v3/fixtures/team/{teamId}'
};

// Cache objesi - performans için
const statsCache = new Map();
const CACHE_DURATION = 5 * 60 * 1000; // 5 dakika cache

// Cache key oluştur
function getCacheKey(teamId, teamName) {
    return `${teamId}_${teamName}`;
}

// Prefetch fonksiyonu - mouse hover'da çalışacak
window.prefetchTeamStats = async function(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Cache key'leri oluştur
    const homeCacheKey = getCacheKey(homeTeamId, homeTeamName);
    const awayCacheKey = getCacheKey(awayTeamId, awayTeamName);
    
    // Zaten cache'de varsa çağrı yapma
    const homeCache = statsCache.get(homeCacheKey);
    const awayCache = statsCache.get(awayCacheKey);
    
    const promises = [];
    
    // Cache'de yoksa arka planda yükle
    if (!homeCache || (Date.now() - homeCache.timestamp > CACHE_DURATION)) {
        promises.push(
            fetchTeamStats(homeTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName)
                .then(data => {
                    statsCache.set(homeCacheKey, { data, timestamp: Date.now() });
                    console.log(`Prefetch: ${homeTeamName} verileri cache'e alındı`);
                })
                .catch(err => console.log(`Prefetch hatası: ${err.message}`))
        );
    }
    
    if (!awayCache || (Date.now() - awayCache.timestamp > CACHE_DURATION)) {
        promises.push(
            fetchTeamStats(awayTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName)
                .then(data => {
                    statsCache.set(awayCacheKey, { data, timestamp: Date.now() });
                    console.log(`Prefetch: ${awayTeamName} verileri cache'e alındı`);
                })
                .catch(err => console.log(`Prefetch hatası: ${err.message}`))
        );
    }
    
    // Arka planda yükle, sonucu bekleme
    if (promises.length > 0) {
        Promise.all(promises);
    }
};

// Modal'ı temizle
function clearTeamStatsModal() {
    // Tüm tab içeriklerini temizle
    const homeStats = document.getElementById('homeTeamStats');
    const awayStats = document.getElementById('awayTeamStats');
    const comparisonStats = document.getElementById('comparisonStats');
    
    if (homeStats) homeStats.innerHTML = '';
    if (awayStats) awayStats.innerHTML = '';
    if (comparisonStats) comparisonStats.innerHTML = '';
    
    // Loading durumlarını sıfırla
    const homeLoading = document.getElementById('homeTeamStatsLoading');
    const awayLoading = document.getElementById('awayTeamStatsLoading');
    const comparisonLoading = document.getElementById('comparisonLoading');
    
    if (homeLoading) {
        homeLoading.style.display = 'block';
        if (homeStats) homeStats.style.display = 'none';
    }
    if (awayLoading) {
        awayLoading.style.display = 'block';
        if (awayStats) awayStats.style.display = 'none';
    }
    if (comparisonLoading) {
        comparisonLoading.style.display = 'block';
        if (comparisonStats) comparisonStats.style.display = 'none';
    }
    
    // İlk tab'ı aktif yap
    const homeTab = document.getElementById('home-team-tab');
    const awayTab = document.getElementById('away-team-tab');
    const h2hTab = document.getElementById('head-to-head-tab');
    
    if (homeTab) {
        homeTab.classList.add('active');
        homeTab.setAttribute('aria-selected', 'true');
    }
    if (awayTab) {
        awayTab.classList.remove('active');
        awayTab.setAttribute('aria-selected', 'false');
    }
    if (h2hTab) {
        h2hTab.classList.remove('active');
        h2hTab.setAttribute('aria-selected', 'false');
    }
    
    // Tab içeriklerini de sıfırla
    const homeContent = document.getElementById('home-team-content');
    const awayContent = document.getElementById('away-team-content');
    const h2hContent = document.getElementById('head-to-head-content');
    
    if (homeContent) {
        homeContent.classList.add('show', 'active');
    }
    if (awayContent) {
        awayContent.classList.remove('show', 'active');
    }
    if (h2hContent) {
        h2hContent.classList.remove('show', 'active');
    }
}

// Takım istatistiklerini gösteren modern popup - V2
function showTeamStats(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Event propagation'ı durdur
    if (event) event.stopPropagation();
    
    // Eski modalı kaldır
    const existingModal = document.getElementById('teamStatsModal');
    if (existingModal) {
        const oldInstance = bootstrap.Modal.getInstance(existingModal);
        if (oldInstance) oldInstance.dispose();
        existingModal.remove();
    }
    
    // Modern popup modalı için HTML hazırla
    const modalHTML = `
    <div class="modal fade team-stats-modal-v2" id="teamStatsModal" tabindex="-1" aria-labelledby="teamStatsModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="teamStatsModalLabel">
                        <i class="fas fa-chart-line"></i>
                        Takım Detaylı İstatistikleri
                    </h5>
                    <button type="button" class="btn-close-modern" data-bs-dismiss="modal" aria-label="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <!-- Modern Tab Navigation -->
                    <div class="nav-tabs-modern" id="teamStatsTabs" role="tablist">
                        <button class="nav-tab-item active" id="home-team-tab" data-tab="home" type="button" role="tab">
                            <i class="fas fa-home"></i>
                            <span id="homeTeamTabName">${homeTeamName || 'Ev Sahibi'}</span>
                        </button>
                        <button class="nav-tab-item" id="away-team-tab" data-tab="away" type="button" role="tab">
                            <i class="fas fa-plane"></i>
                            <span id="awayTeamTabName">${awayTeamName || 'Deplasman'}</span>
                        </button>
                        <button class="nav-tab-item" id="head-to-head-tab" data-tab="comparison" type="button" role="tab">
                            <i class="fas fa-balance-scale"></i>
                            <span>Karşılaştırma</span>
                        </button>
                    </div>
                    
                    <!-- Tab Content Panels -->
                    <div class="tab-content-panels">
                        <!-- Home Team Panel -->
                        <div class="tab-panel active" id="home-team-content" role="tabpanel">
                            <div id="homeTeamStatsLoading" class="loading-spinner">
                                <div class="spinner-ring"></div>
                                <p class="loading-text">İstatistikler yükleniyor...</p>
                            </div>
                            <div id="homeTeamStats" class="stats-container-v2" style="display: none;"></div>
                        </div>
                        
                        <!-- Away Team Panel -->
                        <div class="tab-panel" id="away-team-content" role="tabpanel" style="display: none;">
                            <div id="awayTeamStatsLoading" class="loading-spinner">
                                <div class="spinner-ring"></div>
                                <p class="loading-text">İstatistikler yükleniyor...</p>
                            </div>
                            <div id="awayTeamStats" class="stats-container-v2" style="display: none;"></div>
                        </div>
                        
                        <!-- Comparison Panel -->
                        <div class="tab-panel" id="head-to-head-content" role="tabpanel" style="display: none;">
                            <div id="comparisonLoading" class="loading-spinner">
                                <div class="spinner-ring"></div>
                                <p class="loading-text">Karşılaştırma hazırlanıyor...</p>
                            </div>
                            <div id="comparisonStats" class="stats-container-v2" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>`;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Tab switching logic
    const tabButtons = document.querySelectorAll('.nav-tab-item');
    const tabPanels = document.querySelectorAll('.tab-panel');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Update buttons
            tabButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Update panels
            tabPanels.forEach(panel => {
                panel.style.display = 'none';
            });
            
            if (targetTab === 'home') {
                document.getElementById('home-team-content').style.display = 'block';
            } else if (targetTab === 'away') {
                document.getElementById('away-team-content').style.display = 'block';
            } else if (targetTab === 'comparison') {
                document.getElementById('head-to-head-content').style.display = 'block';
            }
        });
    });
    
    // Modal elementlerini al veya güncelle
    const modalElement = document.getElementById('teamStatsModal');
    
    // Var olan modal instance'ı varsa dispose et
    const currentModalInstance = bootstrap.Modal.getInstance(modalElement);
    if (currentModalInstance) {
        currentModalInstance.dispose();
    }
    
    // Önce eski verileri temizle
    clearTeamStatsModal();
    
    // Tab isimlerini güncelle
    document.getElementById('homeTeamTabName').textContent = homeTeamName;
    document.getElementById('awayTeamTabName').textContent = awayTeamName;
    
    // Yeni modal instance oluştur
    const modal = new bootstrap.Modal(modalElement, {
        backdrop: 'static',
        keyboard: true
    });
    
    // Modal kapatıldığında temizle
    modalElement.addEventListener('hidden.bs.modal', function handleModalHidden() {
        clearTeamStatsModal();
        // Event listener'ı kaldır
        modalElement.removeEventListener('hidden.bs.modal', handleModalHidden);
    });
    
    // Modalı göster
    modal.show();
    
    // İstatistikleri yükle
    loadTeamStatistics(homeTeamId, awayTeamId, homeTeamName, awayTeamName);
}

// İstatistikleri yükleyen ana fonksiyon - Optimized
async function loadTeamStatistics(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    const homeTeamStatsLoading = document.getElementById('homeTeamStatsLoading');
    const awayTeamStatsLoading = document.getElementById('awayTeamStatsLoading');
    const comparisonLoading = document.getElementById('comparisonLoading');
    const homeTeamStats = document.getElementById('homeTeamStats');
    const awayTeamStats = document.getElementById('awayTeamStats');
    const comparisonStats = document.getElementById('comparisonStats');
    
    try {
        // Tab değişiminde lazy loading için event listener ekle (V2 modern tabs)
        const tabButtons = document.querySelectorAll('#teamStatsTabs .nav-tab-item');
        let homeLoaded = false;
        let awayLoaded = false;
        let comparisonLoaded = false;
        
        // İlk tab (ev sahibi) hemen yükle
        const startTime = Date.now();
        
        // Cache'den kontrol et
        const homeCacheKey = getCacheKey(homeTeamId, homeTeamName);
        const awayCacheKey = getCacheKey(awayTeamId, awayTeamName);
        
        let homeStats = null;
        let awayStats = null;
        
        // Cache'de varsa hemen kullan
        const homeCache = statsCache.get(homeCacheKey);
        const awayCache = statsCache.get(awayCacheKey);
        
        if (homeCache && (Date.now() - homeCache.timestamp < CACHE_DURATION)) {
            homeStats = homeCache.data;
            console.log(`Ev sahibi istatistikleri cache'den yüklendi (${Date.now() - startTime}ms)`);
        }
        
        if (awayCache && (Date.now() - awayCache.timestamp < CACHE_DURATION)) {
            awayStats = awayCache.data;
            console.log(`Deplasman istatistikleri cache'den yüklendi (${Date.now() - startTime}ms)`);
        }
        
        // Paralel API çağrıları - sadece cache'de olmayanlar için
        const promises = [];
        
        if (!homeStats) {
            promises.push(
                fetchTeamStats(homeTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName)
                    .then(data => {
                        homeStats = data;
                        statsCache.set(homeCacheKey, { data, timestamp: Date.now() });
                        return data;
                    })
            );
        }
        
        if (!awayStats) {
            promises.push(
                fetchTeamStats(awayTeamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName)
                    .then(data => {
                        awayStats = data;
                        statsCache.set(awayCacheKey, { data, timestamp: Date.now() });
                        return data;
                    })
            );
        }
        
        // Sadece gerekli olanları bekle
        if (promises.length > 0) {
            await Promise.all(promises);
        }
        
        console.log(`Tüm istatistikler yüklendi (${Date.now() - startTime}ms)`);
        
        // İstatistikleri modern şekilde göster
        displayModernTeamStats(homeStats, homeTeamStats, homeTeamStatsLoading, homeTeamName, 'home');
        homeLoaded = true;
        
        // Tab değişim event'lerini dinle - lazy loading için (V2 modern tabs)
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const target = this.getAttribute('data-tab');
                
                if (target === 'away' && !awayLoaded) {
                    displayModernTeamStats(awayStats, awayTeamStats, awayTeamStatsLoading, awayTeamName, 'away');
                    awayLoaded = true;
                } else if (target === 'comparison' && !comparisonLoaded) {
                    displayComparison(homeStats, awayStats, comparisonStats, comparisonLoading, homeTeamName, awayTeamName);
                    comparisonLoaded = true;
                }
            });
        });
        
        // Arka planda diğer tab'ları da yükle
        setTimeout(() => {
            if (!awayLoaded) {
                displayModernTeamStats(awayStats, awayTeamStats, awayTeamStatsLoading, awayTeamName, 'away');
                awayLoaded = true;
            }
            if (!comparisonLoaded) {
                displayComparison(homeStats, awayStats, comparisonStats, comparisonLoading, homeTeamName, awayTeamName);
                comparisonLoaded = true;
            }
        }, 100);
        
    } catch (error) {
        console.error('Takım istatistikleri yüklenirken hata:', error);
        homeTeamStatsLoading.innerHTML = `<div class="alert alert-danger">İstatistikler yüklenemedi: ${error.message}</div>`;
        awayTeamStatsLoading.innerHTML = `<div class="alert alert-danger">İstatistikler yüklenemedi: ${error.message}</div>`;
    }
}

// Takım istatistiklerini API'den çek - Timeout ile
async function fetchTeamStats(teamId, homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    try {
        // 1. YÖNTEM: Doğrudan tahmin API'sinden veri çekmek (çalıştığını biliyoruz)
        try {
            console.log(`Takım ${teamId} için tahmin API'sinden bilgileri alıyoruz...`);
            // Takım adını parametre olarak alıyoruz
            const teamNameParam = teamId === homeTeamId ? homeTeamName : awayTeamName;
            
            // Timeout ile birlikte API çağrısı
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 saniye timeout
            
            const predictionResponse = await fetch(`/api/predict-match/${teamId}/${teamId}?home_name=${encodeURIComponent(teamNameParam)}&away_name=${encodeURIComponent(teamNameParam)}`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (predictionResponse.ok) {
                const predictionData = await predictionResponse.json();
                console.log("Tahmin API'sinden veri başarıyla alındı:", predictionData);
                
                // Ev sahibi takım formu verileri
                if (predictionData && predictionData.home_team && predictionData.home_team.form) {
                    // Takım adını API'den gelen veriden veya modal başlığından al
                    const teamName = predictionData.home_team.name || teamNameParam || `Takım ${teamId}`;
                    const formData = predictionData.home_team.form;
                    
                    // Takımın detaylı form verilerinden maç bilgilerini çıkar
                    if (formData.detailed_data && formData.detailed_data.all) {
                        const matches = formData.detailed_data.all;
                        const formattedMatches = [];
                        
                        matches.forEach(match => {
                            // Takım adını ve rakip takım adını birlikte göster
                            formattedMatches.push({
                                date: match.date || "",
                                match: `${match.is_home ? teamName : (match.opponent || "Rakip")} vs ${match.is_home ? (match.opponent || "Rakip") : teamName}`,
                                score: `${match.goals_scored} - ${match.goals_conceded}`
                            });
                        });
                        
                        if (formattedMatches.length > 0) {
                            console.log(`Tahmin API'sinden ${formattedMatches.length} maç verisi alındı`);
                            return formattedMatches;
                        }
                    }
                }
            }
        } catch (predictionError) {
            console.error("Tahmin API'sinden veri çekerken hata:", predictionError);
        }
        
        // 2. YÖNTEM: Takım istatistikleri API'sini kullanmak
        try {
            console.log(`Takım ${teamId} için istatistik API'sinden bilgileri alıyoruz...`);
            const statsResponse = await fetch(`/api/v3/team-stats/${teamId}`);
            
            if (statsResponse.ok) {
                const statsData = await statsResponse.json();
                if (statsData && statsData.length > 0) {
                    console.log(`İstatistik API'sinden ${statsData.length} maç verisi alındı`);
                    return statsData;
                }
            }
        } catch (statsError) {
            console.error("İstatistik API'sinden veri çekerken hata:", statsError);
        }
        
        // 3. YÖNTEM: Tahmin sayfasını yüklemeyi dene ve HTML'den veriler çıkar
        try {
            console.log(`Takım ${teamId} için HTML sayfasından bilgileri çıkarmayı deniyoruz...`);
            const response = await fetch(`/predict-match/${teamId}/9999`);
            
            if (response.ok) {
                const html = await response.text();
                const teamName = extractTeamNameFromHTML(html, teamId);
                console.log(`HTML'den takım adı: ${teamName}`);
                
                // Takım adını bulduysak minimum veriler oluştur
                const teamMatches = [];
                const today = new Date();
                
                for (let i = 0; i < 5; i++) {
                    const pastDate = new Date(today);
                    pastDate.setDate(today.getDate() - (i * 7));
                    const dateStr = pastDate.toISOString().split('T')[0];
                    
                    teamMatches.push({
                        date: dateStr,
                        match: `${teamName} - Son ${i+1} Maç`,
                        score: "Sonuç bilgisi bulunamadı"
                    });
                }
                
                console.log(`HTML sayfasından ${teamMatches.length} geçici maç verisi oluşturuldu`);
                return teamMatches;
            }
        } catch (htmlError) {
            console.error("HTML sayfasından veri çıkarmada hata:", htmlError);
        }
        
        // HTML'den takım adını çıkarma fonksiyonu
        function extractTeamNameFromHTML(html, teamId) {
            try {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, "text/html");
                
                // Başlıktan takım adını çıkarmaya çalış
                const titleElement = doc.querySelector('title');
                if (titleElement && titleElement.textContent) {
                    const titleText = titleElement.textContent;
                    
                    // Başlık genellikle "Ev Sahibi vs Deplasman" formatındadır
                    const vsIndex = titleText.indexOf(' vs ');
                    if (vsIndex > 0) {
                        return titleText.substring(0, vsIndex).trim();
                    }
                }
                
                // H4 elementlerinden takım adını çıkarmaya çalış
                const h4Elements = doc.querySelectorAll('h4');
                for (let h4 of h4Elements) {
                    if (h4.textContent && h4.textContent.trim() !== '') {
                        return h4.textContent.trim();
                    }
                }
                
                // Varsayılan takım adı
                return `Takım ${teamId}`;
            } catch (e) {
                console.error("HTML'den takım adı çıkarmada hata:", e);
                return `Takım ${teamId}`;
            }
        }
        
        // Orijinal API ile deneyelim
        try {
            const response = await fetch(`/api/v3/fixtures/team/${teamId}`);
            if (response.ok) {
                const data = await response.json();
                if (data && data.length > 0) {
                    return data;
                }
            }
        } catch (apiError) {
            console.error("Birincil API ile veri alınamadı:", apiError);
        }
        
        // Yedek API ile deneyelim
        try {
            // Takım adını parametre olarak gönderelim
            const teamNameParam = teamId === homeTeamId ? homeTeamName : awayTeamName;
            const backupResponse = await fetch(`/api/team-matches/${teamId}?team_name=${encodeURIComponent(teamNameParam)}&stats=true`);
            if (backupResponse.ok) {
                const backupData = await backupResponse.json();
                if (backupData && backupData.matches && Array.isArray(backupData.matches)) {
                    const teamMatches = [];
                    backupData.matches.forEach(match => {
                        teamMatches.push({
                            date: match.date || '',
                            match: match.match || '',
                            score: match.score || ''
                        });
                    });
                    return teamMatches;
                }
            }
        } catch (backupError) {
            console.error(`Yedek API ile takım verileri alınamadı:`, backupError);
        }
        
        // Tüm API'ler başarısız olursa minimal veri döndür
        const teamMatches = [];
        const today = new Date();
        
        for (let i = 0; i < 3; i++) {
            const pastDate = new Date(today);
            pastDate.setDate(today.getDate() - (i * 7));
            const dateStr = pastDate.toISOString().split('T')[0];
            
            teamMatches.push({
                date: dateStr,
                match: `Takım ${teamId} - Maç bilgisi`,
                score: "API'den veri alınamadı"
            });
        }
        
        return teamMatches;
    } catch (error) {
        console.error(`Takım (${teamId}) istatistikleri alınırken hata:`, error);
        
        // Minimal veri döndür
        return [
            {
                date: new Date().toISOString().split('T')[0],
                match: `Takım ${teamId} - Veri bulunamadı`,
                score: "Hata oluştu"
            }
        ];
    }
}

// Modern takım istatistiklerini göster - V2
function displayModernTeamStats(stats, container, loadingElement, teamName, teamType) {
    if (!stats || stats.length === 0) {
        container.innerHTML = `
            <div class="team-header-card">
                <div class="team-name-title">${teamName}</div>
                <div class="team-subtitle" style="color: #f59e0b;">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Bu takım için veri bulunamadı
                </div>
            </div>`;
        loadingElement.style.display = 'none';
        container.style.display = 'block';
        return;
    }
    
    const wins = countResults(stats, 'win', teamName);
    const draws = countResults(stats, 'draw', teamName);
    const losses = countResults(stats, 'loss', teamName);
    
    let html = `
        <div class="team-header-card">
            <div class="team-name-title">${teamName}</div>
            <div class="team-subtitle">
                <i class="fas fa-futbol me-1"></i> Son 5 Maç Performansı
            </div>
        </div>
        
        <div class="matches-grid-v2">`;
    
    stats.slice(0, 5).forEach((match, index) => {
        const matchDate = match.date || '';
        const matchInfo = match.match || 'Maç bilgisi yok';
        const score = match.score || '-';
        
        let result = 'draw';
        let resultText = 'Berabere';
        
        if (score && score !== '-' && score !== 'Sonuç bilgisi bulunamadı') {
            const scoreParts = score.split('-');
            if (scoreParts.length === 2) {
                const homeScore = parseInt(scoreParts[0]);
                const awayScore = parseInt(scoreParts[1]);
                const isHome = matchInfo.toLowerCase().indexOf(teamName.toLowerCase()) < 
                               matchInfo.toLowerCase().indexOf('vs');
                
                if (isHome) {
                    if (homeScore > awayScore) { result = 'win'; resultText = 'Galibiyet'; }
                    else if (homeScore < awayScore) { result = 'loss'; resultText = 'Mağlubiyet'; }
                } else {
                    if (awayScore > homeScore) { result = 'win'; resultText = 'Galibiyet'; }
                    else if (awayScore < homeScore) { result = 'loss'; resultText = 'Mağlubiyet'; }
                }
            }
        }
        
        html += `
            <div class="match-card-v2 ${result}">
                <div class="match-card-header">
                    <span class="result-badge ${result}">${resultText}</span>
                    <span class="match-date">
                        <i class="fas fa-calendar-alt"></i>
                        ${matchDate}
                    </span>
                </div>
                <div class="match-teams-v2">${matchInfo}</div>
                <div class="match-score-v2">
                    <i class="fas fa-futbol" style="font-size: 0.8rem; opacity: 0.7;"></i>
                    ${score}
                </div>
            </div>`;
    });
    
    html += `
        </div>
        
        <div class="stats-summary-v2">
            <div class="summary-title">
                <i class="fas fa-chart-pie me-2"></i>
                İstatistik Özeti
            </div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-value win">${wins}</div>
                    <div class="summary-label">Galibiyet</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value draw">${draws}</div>
                    <div class="summary-label">Beraberlik</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value loss">${losses}</div>
                    <div class="summary-label">Mağlubiyet</div>
                </div>
            </div>
        </div>`;
    
    container.innerHTML = html;
    loadingElement.style.display = 'none';
    container.style.display = 'block';
}

// Sonuçları say - takım adına göre doğru hesaplama yap
function countResults(stats, resultType, teamName) {
    let count = 0;
    stats.slice(0, 5).forEach(match => {
        const score = match.score || '-';
        const matchInfo = match.match || '';
        
        if (score && score !== '-' && score !== 'Sonuç bilgisi bulunamadı') {
            const scoreParts = score.split('-');
            if (scoreParts.length === 2) {
                const homeScore = parseInt(scoreParts[0]);
                const awayScore = parseInt(scoreParts[1]);
                
                // Takımın ev sahibi mi deplasman mı olduğunu belirle
                const isHome = matchInfo.toLowerCase().indexOf(teamName.toLowerCase()) < 
                               matchInfo.toLowerCase().indexOf('vs');
                
                let matchResult = '';
                if (homeScore === awayScore) {
                    matchResult = 'draw';
                } else if (isHome) {
                    matchResult = homeScore > awayScore ? 'win' : 'loss';
                } else {
                    matchResult = awayScore > homeScore ? 'win' : 'loss';
                }
                
                if (resultType === matchResult) count++;
            }
        }
    });
    return count;
}

// Karşılaştırma görünümü - V2 Modern Tasarım
function displayComparison(homeStats, awayStats, container, loadingElement, homeTeamName, awayTeamName) {
    const homeWins = countResults(homeStats, 'win', homeTeamName);
    const homeDraws = countResults(homeStats, 'draw', homeTeamName);
    const homeLosses = countResults(homeStats, 'loss', homeTeamName);
    
    const awayWins = countResults(awayStats, 'win', awayTeamName);
    const awayDraws = countResults(awayStats, 'draw', awayTeamName);
    const awayLosses = countResults(awayStats, 'loss', awayTeamName);
    
    const homeTotal = homeWins + homeDraws + homeLosses || 1;
    const awayTotal = awayWins + awayDraws + awayLosses || 1;
    
    const homeWinRate = Math.round((homeWins / homeTotal) * 100);
    const awayWinRate = Math.round((awayWins / awayTotal) * 100);
    
    let homeGoalsScored = 0, homeGoalsConceded = 0;
    let awayGoalsScored = 0, awayGoalsConceded = 0;
    
    homeStats.slice(0, 5).forEach(match => {
        if (match.score && match.score !== '-') {
            const parts = match.score.split('-');
            if (parts.length === 2) {
                const isHome = match.match.toLowerCase().indexOf(homeTeamName.toLowerCase()) < 
                               match.match.toLowerCase().indexOf('vs');
                if (isHome) {
                    homeGoalsScored += parseInt(parts[0]) || 0;
                    homeGoalsConceded += parseInt(parts[1]) || 0;
                } else {
                    homeGoalsScored += parseInt(parts[1]) || 0;
                    homeGoalsConceded += parseInt(parts[0]) || 0;
                }
            }
        }
    });
    
    awayStats.slice(0, 5).forEach(match => {
        if (match.score && match.score !== '-') {
            const parts = match.score.split('-');
            if (parts.length === 2) {
                const isHome = match.match.toLowerCase().indexOf(awayTeamName.toLowerCase()) < 
                               match.match.toLowerCase().indexOf('vs');
                if (isHome) {
                    awayGoalsScored += parseInt(parts[0]) || 0;
                    awayGoalsConceded += parseInt(parts[1]) || 0;
                } else {
                    awayGoalsScored += parseInt(parts[1]) || 0;
                    awayGoalsConceded += parseInt(parts[0]) || 0;
                }
            }
        }
    });
    
    const homeAvgScored = (homeGoalsScored / (homeTotal || 1)).toFixed(1);
    const homeAvgConceded = (homeGoalsConceded / (homeTotal || 1)).toFixed(1);
    const awayAvgScored = (awayGoalsScored / (awayTotal || 1)).toFixed(1);
    const awayAvgConceded = (awayGoalsConceded / (awayTotal || 1)).toFixed(1);
    
    // Karşılaştırma bar hesaplamaları
    const totalWinRate = homeWinRate + awayWinRate || 1;
    const homeWinRatePercent = Math.round((homeWinRate / totalWinRate) * 100);
    const awayWinRatePercent = 100 - homeWinRatePercent;
    
    const totalScored = parseFloat(homeAvgScored) + parseFloat(awayAvgScored) || 1;
    const homeScoredPercent = Math.round((parseFloat(homeAvgScored) / totalScored) * 100);
    const awayScoredPercent = 100 - homeScoredPercent;
    
    const html = `
        <div class="comparison-container">
            <!-- Teams Header -->
            <div class="teams-comparison-header">
                <div class="team-card home">
                    <div class="team-card-name">${homeTeamName}</div>
                    <div class="team-form-badges">
                        <span class="form-badge w">${homeWins}G</span>
                        <span class="form-badge d">${homeDraws}B</span>
                        <span class="form-badge l">${homeLosses}M</span>
                    </div>
                    <div class="win-rate-circle">
                        <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="rgba(99, 102, 241, 0.2)" stroke-width="3"/>
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#6366f1" stroke-width="3"
                                stroke-dasharray="${homeWinRate}, 100"/>
                        </svg>
                        <div class="win-rate-value">${homeWinRate}%</div>
                    </div>
                </div>
                
                <div class="vs-badge">VS</div>
                
                <div class="team-card away">
                    <div class="team-card-name">${awayTeamName}</div>
                    <div class="team-form-badges">
                        <span class="form-badge w">${awayWins}G</span>
                        <span class="form-badge d">${awayDraws}B</span>
                        <span class="form-badge l">${awayLosses}M</span>
                    </div>
                    <div class="win-rate-circle">
                        <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="rgba(236, 72, 153, 0.2)" stroke-width="3"/>
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#ec4899" stroke-width="3"
                                stroke-dasharray="${awayWinRate}, 100"/>
                        </svg>
                        <div class="win-rate-value">${awayWinRate}%</div>
                    </div>
                </div>
            </div>
            
            <!-- Comparison Bars -->
            <div class="comparison-section">
                <div class="comparison-section-title">
                    <i class="fas fa-chart-bar"></i>
                    Performans Karşılaştırması
                </div>
                
                <div class="comparison-bar-item">
                    <div class="bar-label">
                        <span class="bar-label-text">Kazanma Oranı</span>
                        <span class="bar-label-values">${homeWinRate}% - ${awayWinRate}%</span>
                    </div>
                    <div class="comparison-bar-track">
                        <div class="bar-fill-home" style="width: ${homeWinRatePercent}%;"></div>
                        <div class="bar-fill-away" style="width: ${awayWinRatePercent}%;"></div>
                    </div>
                </div>
                
                <div class="comparison-bar-item">
                    <div class="bar-label">
                        <span class="bar-label-text">Gol Ortalaması</span>
                        <span class="bar-label-values">${homeAvgScored} - ${awayAvgScored} gol/maç</span>
                    </div>
                    <div class="comparison-bar-track">
                        <div class="bar-fill-home" style="width: ${homeScoredPercent}%;"></div>
                        <div class="bar-fill-away" style="width: ${awayScoredPercent}%;"></div>
                    </div>
                </div>
            </div>
            
            <!-- Goal Stats -->
            <div class="goal-stats-grid">
                <div class="goal-stat-card home">
                    <div style="color: #a5b4fc; font-size: 0.75rem; margin-bottom: 0.75rem; text-transform: uppercase;">${homeTeamName}</div>
                    <div class="goal-stat-row">
                        <div class="goal-stat-item">
                            <div class="goal-icon scored"><i class="fas fa-bullseye"></i></div>
                            <div class="goal-value">${homeAvgScored}</div>
                            <div class="goal-label">Attığı/Maç</div>
                        </div>
                        <div class="goal-stat-item">
                            <div class="goal-icon conceded"><i class="fas fa-shield-alt"></i></div>
                            <div class="goal-value">${homeAvgConceded}</div>
                            <div class="goal-label">Yediği/Maç</div>
                        </div>
                    </div>
                </div>
                <div class="goal-stat-card away">
                    <div style="color: #f9a8d4; font-size: 0.75rem; margin-bottom: 0.75rem; text-transform: uppercase;">${awayTeamName}</div>
                    <div class="goal-stat-row">
                        <div class="goal-stat-item">
                            <div class="goal-icon scored"><i class="fas fa-bullseye"></i></div>
                            <div class="goal-value">${awayAvgScored}</div>
                            <div class="goal-label">Attığı/Maç</div>
                        </div>
                        <div class="goal-stat-item">
                            <div class="goal-icon conceded"><i class="fas fa-shield-alt"></i></div>
                            <div class="goal-value">${awayAvgConceded}</div>
                            <div class="goal-label">Yediği/Maç</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
    
    container.innerHTML = html;
    loadingElement.style.display = 'none';
    container.style.display = 'block';
}

// Eski displayTeamStats fonksiyonu (geriye uyumluluk için)
function displayTeamStats(stats, container, loadingElement) {
    // Her durumda bir şeyler göster
    loadingElement.style.display = 'none';
    container.style.display = 'block';
    
    if (!stats || !stats.length) {
        // Veri yoksa bilgi ver
        container.innerHTML = `
            <div class="alert alert-warning bg-dark text-warning border-dark">
                <p>Bu takım için istatistik bulunamadı.</p>
                <p>Olası nedenler:</p>
                <ul>
                    <li>Takım son dönemde maç oynamamış olabilir</li>
                    <li>API veritabanında takım bilgisi eksik olabilir</li>
                    <li>Takım ID'si API ile uyumlu değil</li>
                </ul>
                <p>Farklı bir takım seçmeyi deneyin.</p>
            </div>`;
        return;
    }
    
    // En az bazı maçlar gösterilebiliyorsa, HTML oluştur
    let html = '<div class="list-group bg-dark">';
    stats.forEach(match => {
        html += `
            <div class="list-group-item bg-dark text-light border-secondary">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <div><small class="text-info">${match.date || ''}</small></div>
                </div>
                <div class="text-center">
                    <span class="match-teams text-light">${match.match || ''}</span>
                    <br>
                    <strong class="match-score text-warning">${match.score || ''}</strong>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

// Global scope'a ekle
window.showTeamStats = showTeamStats;

// Hover event listener'ları ekle - prefetch için
document.addEventListener('DOMContentLoaded', function() {
    // Match item'lara hover listener ekle
    document.addEventListener('mouseover', function(e) {
        const matchItem = e.target.closest('.match-item');
        if (matchItem && !matchItem.dataset.prefetched) {
            const homeTeamId = matchItem.dataset.homeId;
            const awayTeamId = matchItem.dataset.awayId;
            const homeTeamName = matchItem.dataset.homeName;
            const awayTeamName = matchItem.dataset.awayName;
            
            if (homeTeamId && awayTeamId && homeTeamName && awayTeamName) {
                // Prefetch yap
                window.prefetchTeamStats(homeTeamId, awayTeamId, homeTeamName, awayTeamName);
                // İşaretleyelim ki tekrar prefetch yapmasın
                matchItem.dataset.prefetched = 'true';
            }
        }
    });
});

// Sayfa yüklendiğinde Modern Takım İstatistikleri mesajı
console.log('Modern Takım İstatistikleri modülü yüklendi');

// API parametrelerini konsola yazdır (debugging için)
console.log('API Endpoints:', TEAM_STATS_API);