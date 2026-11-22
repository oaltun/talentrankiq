export const LANGUAGES = ["en", "tr"] as const

export type Language = (typeof LANGUAGES)[number]

export const DEFAULT_LANGUAGE: Language = "en"

export const LANGUAGE_STORAGE_KEY = "language"

export const LANGUAGE_NAMES: Record<Language, string> = {
  en: "English",
  tr: "Türkçe",
}

export type TranslationVars = Record<string, string | number>

export const translations = {
  "app.name": {
    en: "TalentRankIQ",
    tr: "TalentRankIQ",
  },
  "common.dismiss": {
    en: "Dismiss",
    tr: "Kapat",
  },
  "nav.jobs": {
    en: "Jobs",
    tr: "İşler",
  },
  "nav.models": {
    en: "Models",
    tr: "Modeller",
  },
  "nav.settings": {
    en: "Settings",
    tr: "Ayarlar",
  },
  "nav.help": {
    en: "Get Help",
    tr: "Yardım",
  },
  "page.jobs": {
    en: "Jobs",
    tr: "İşler",
  },
  "page.models": {
    en: "Models",
    tr: "Modeller",
  },
  "page.jobDetails": {
    en: "Job Details",
    tr: "İş Detayları",
  },
  "jobs.title": {
    en: "Jobs",
    tr: "İşler",
  },
  "jobs.subtitle": {
    en: "Manage your TalentRankIQ jobs. Add a job to get started.",
    tr: "TalentRankIQ işlerinizi yönetin. Başlamak için bir iş ekleyin.",
  },
  "jobs.addJob": {
    en: "Add Job",
    tr: "İş Ekle",
  },
  "jobs.loading": {
    en: "Loading jobs...",
    tr: "İşler yükleniyor...",
  },
  "jobs.noJobs": {
    en: "No jobs yet.",
    tr: "Henüz iş yok.",
  },
  "jobs.archived": {
    en: "Archived",
    tr: "Arşivlendi",
  },
  "jobs.cvs": {
    en: "{{count}} CVs",
    tr: "{{count}} CV",
  },
  "jobs.failed": {
    en: "{{count}} failed",
    tr: "{{count}} başarısız",
  },
  "jobs.ranked": {
    en: "ranked {{when}}",
    tr: "{{when}} sıralandı",
  },
  "jobs.today": {
    en: "today",
    tr: "bugün",
  },
  "jobs.daysAgo": {
    en: "{{days}}d ago",
    tr: "{{days}} gün önce",
  },
  "jobs.restore": {
    en: "Restore",
    tr: "Geri Yükle",
  },
  "jobs.archive": {
    en: "Archive",
    tr: "Arşivle",
  },
  "jobs.delete": {
    en: "Delete",
    tr: "Sil",
  },
  "jobs.more": {
    en: "More",
    tr: "Diğer",
  },
  "jobs.error.load": {
    en: "Failed to load jobs",
    tr: "İşler yüklenemedi",
  },
  "jobs.error.create": {
    en: "Failed to create job",
    tr: "İş oluşturulamadı",
  },
  "jobs.error.update": {
    en: "Failed to update job",
    tr: "İş güncellenemedi",
  },
  "jobs.error.delete": {
    en: "Failed to delete job",
    tr: "İş silinemedi",
  },
  "models.title": {
    en: "Models",
    tr: "Modeller",
  },
  "models.subtitle": {
    en: "Manage the on-device embedding models used for ranking CVs.",
    tr: "CV'leri sıralamak için kullanılan cihazdaki gömme modellerini yönetin.",
  },
  "models.description": {
    en: 'On-device embedding models from the app\'s model registry. Download, remove, and pick the model used for ranking CVs. The engine starts when you click "Set default" on a downloaded model.',
    tr: 'Cihazda gömme modellerini kullanarak CV\'leri sıralamak için kullanılan modelleri yönetin. Modeli indir, sil ve "Set default" (varsayılan yapılandırmak) üzerine tıkladığınızda motor başlar.',
  },
  "models.noModelsDownloaded": {
    en: "No models downloaded",
    tr: "Model indirilmedi",
  },
  "models.noModelsDescription": {
    en: "Download a model to enable AI ranking. The engine will start when you make a model default.",
    tr: "AI sıralamasını etkinleştirmek için bir model indirin. Motor, bir modeli varsayılan yapınca başlar.",
  },
  "models.noEmbeddersConfigured": {
    en: "No embedders configured in triq.yaml.",
    tr: "triq.yaml'de gömme modelleri yapılandırılmadı.",
  },
  "models.loadingEmbedders": {
    en: "Loading embedders...",
    tr: "Gömme modelleri yükleniyor...",
  },
  "models.error.loadEmbedders": {
    en: "Failed to load embedders",
    tr: "Gömme modelleri yüklenemedi",
  },
  "models.tab.quick": {
    en: "Quick",
    tr: "Hızlı",
  },
  "models.tab.deep": {
    en: "Deep",
    tr: "Derin",
  },
  "models.set_default": {
    en: "Set default",
    tr: "Varsayılan yap",
  },
  "models.default": {
    en: "default",
    tr: "varsayılan",
  },
  "models.remove": {
    en: "Remove",
    tr: "Sil",
  },
  "models.download": {
    en: "Download",
    tr: "İndir",
  },
  "models.ram_required": {
    en: "RAM required",
    tr: "RAM gerekli",
  },
  "models.embedding_dimensions": {
    en: "Embedding dimensions",
    tr: "Gömme boyutları",
  },
  "models.max_context_window": {
    en: "Max context window",
    tr: "Maks. bağlam penceresi",
  },
  "models.disk_size": {
    en: "Disk size",
    tr: "Disk boyutu",
  },
  "models.comparator.section_label": {
    en: "Comparator artifact",
    tr: "Karşılaştırıcı artefakt",
  },
  "models.comparator.pro": {
    en: "Pro",
    tr: "Pro",
  },
  "models.comparator.unlock": {
    en: "Unlock the comparator",
    tr: "Karşılaştırıcıyı aç",
  },
  "models.comparator.sharper_than": {
    en: "Pairwise LLM judging ranks CVs head-to-head — sharper than cosine alone.",
    tr: "CV'leri baş başa LLM yargısıyla sıralar — kosinüs'ten daha kesin.",
  },
  "models.comparator.present": {
    en: "present",
    tr: "var",
  },
  "models.comparator.missing": {
    en: "missing",
    tr: "eksik",
  },
  "models.comparator.file_must_be_named": {
    en: "File must be named",
    tr: "Dosya şu isimle olmalı",
  },
  "models.comparator.get": {
    en: "Get comparator",
    tr: "Karşılaştırıcıyı al",
  },
  "models.comparator.loading": {
    en: "Loading...",
    tr: "Yükleniyor...",
  },
  "models.missing.title": {
    en: "Missing model",
    tr: "Model eksik",
  },
  "models.missing.description.quick": {
    en: "You were redirected from the workspace page to the models page because there is no quick model. Download the quick model on this page.",
    tr: "Çalışma alanı sayfasından modeller sayfasına yönlendirildiniz çünkü hızlı model yok. Hızlı modeli bu sayfadan indirin.",
  },
  "models.missing.description.default": {
    en: "You were redirected from the workspace page to the models page because the default deep model{{model}} is not downloaded. Download the default deep model on this page.",
    tr: "Çalışma alanı sayfasından modeller sayfasına yönlendirildiniz çünkü varsayılan derin model{{model}} indirilmemiş. Varsayılan derin modeli bu sayfadan indirin.",
  },
  "models.missing.description.generic": {
    en: "You were redirected from the workspace page to the models page because there is no model. Download the model on this page.",
    tr: "Çalışma alanı sayfasından modeller sayfasına yönlendirildiniz çünkü model yok. Modeli bu sayfadan indirin.",
  },
  "models.missing.action": {
    en: "Download model",
    tr: "Modeli indir",
  },
  "models.missing.dismiss": {
    en: "Dismiss",
    tr: "Kapat",
  },
  "models.downloaded": {
    en: "Downloaded",
    tr: "İndirildi",
  },
  "models.ready": {
    en: "Ready",
    tr: "Hazır",
  },
  "models.downloading": {
    en: "Downloading...",
    tr: "İndiriliyor...",
  },
  "models.size_unknown": {
    en: "unknown",
    tr: "bilinmiyor",
  },
  "models.quick.loading": {
    en: "Loading quick model...",
    tr: "Hızlı model yükleniyor...",
  },
  "models.quick.loadError": {
    en: "Failed to load quick model",
    tr: "Hızlı model yüklenemedi",
  },
  "models.quick.disabled": {
    en: "Quick stage disabled.",
    tr: "Hızlı aşama devre dışı.",
  },
  "models.quick.downloadFailed": {
    en: "Download failed",
    tr: "İndirme başarısız",
  },
  "models.quick.removeFailed": {
    en: "Remove failed",
    tr: "Kaldırma başarısız",
  },
  "workspace.title": {
    en: "Workspace",
    tr: "Çalışma Alanı",
  },
  "workspace.backToJobs": {
    en: "Back to jobs",
    tr: "İşlere dön",
  },
  "workspace.editTitle": {
    en: "Edit title",
    tr: "Başlığı düzenle",
  },
  "workspace.saveTitle": {
    en: "Save title",
    tr: "Başlığı kaydet",
  },
  "workspace.loadingJob": {
    en: "Loading job...",
    tr: "İş yükleniyor...",
  },
  "workspace.selectJobToStart": {
    en: "Select a job from the Home activity to begin ranking CVs.",
    tr: "CV'leri sıralamaya başlamak için Ana ekrandan bir iş seçin.",
  },
  "workspace.jobNotFound": {
    en: "Job not found",
    tr: "İş bulunamadı",
  },
  "workspace.jobNotFoundDescription": {
    en: "This job may have been deleted or never existed.",
    tr: "Bu iş silinmiş ya da hiç var olmayabilir.",
  },
  "workspace.redirectingToJobs": {
    en: "Redirecting to Jobs in 3 seconds...",
    tr: "3 saniye içinde İşler sayfasına yönlendiriliyorsunuz...",
  },
  "workspace.errorLoadingJob": {
    en: "Failed to load job",
    tr: "İş yüklenemedi",
  },
  "workspace.jobAd": {
    en: "Job Ad",
    tr: "İş İlanı",
  },
  "workspace.autoCreated": {
    en: "Auto-created",
    tr: "Otomatik oluşturuldu",
  },
  "workspace.cvWatchDirectory": {
    en: "CV watch directory",
    tr: "CV izleme dizini",
  },
  "workspace.enterCvDirPath": {
    en: "Enter CV directory path...",
    tr: "CV dizin yolunu girin...",
  },
  "workspace.changeDirectory": {
    en: "Change directory",
    tr: "Dizini değiştir",
  },
  "workspace.openInFileManager": {
    en: "Open in file manager",
    tr: "Dosya yöneticisinde aç",
  },
  "workspace.progress": {
    en: "Progress",
    tr: "İlerleme",
  },
  "workspace.progressQuickEmbedded": {
    en: "Quick embedded",
    tr: "Hızlı gömülü",
  },
  "workspace.progressDeepEmbedded": {
    en: "Deep embedded",
    tr: "Derin gömülü",
  },
  "workspace.progressCompared": {
    en: "Compared pairs",
    tr: "Karşılaştırılan çiftler",
  },
  "workspace.progressQuickSimilarity": {
    en: "Quick similarity",
    tr: "Hızlı benzerlik",
  },
  "workspace.progressDeepSimilarity": {
    en: "Deep similarity",
    tr: "Derin benzerlik",
  },
  "workspace.enterJobAdPath": {
    en: "Enter job ad path...",
    tr: "İş ilanı yolu girin...",
  },
  "workspace.additionalCriteria": {
    en: "Additional Criteria",
    tr: "Ek Kriterler",
  },
  "workspace.rubric": {
    en: "Rubric",
    tr: "Değerlendirme Ölçeği",
  },
  "workspace.capacity": {
    en: "Capacity",
    tr: "Kapasite",
  },
  "workspace.weight": {
    en: "Weight",
    tr: "Ağırlık",
  },
  "workspace.points": {
    en: "pts",
    tr: "p",
  },
  "workspace.candidates": {
    en: "Candidates",
    tr: "Adaylar",
  },
  "workspace.searchCandidates": {
    en: "Search candidates...",
    tr: "Adayları ara...",
  },
  "workspace.tableView": {
    en: "Table view",
    tr: "Tablo görünümü",
  },
  "workspace.cardView": {
    en: "Card view",
    tr: "Kart görünümü",
  },
  "workspace.clearFeedback": {
    en: "Clear Feedback",
    tr: "Geri Bildirimleri Temizle",
  },
  "workspace.rank": {
    en: "Rank",
    tr: "Sıra",
  },
  "workspace.candidate": {
    en: "Candidate",
    tr: "Aday",
  },
  "workspace.total": {
    en: "Total",
    tr: "Toplam",
  },
  "workspace.like": {
    en: "Like",
    tr: "Beğen",
  },
  "workspace.unlike": {
    en: "Unlike",
    tr: "Beğenmeyi Kaldır",
  },
  "workspace.viewDetails": {
    en: "View details",
    tr: "Detayları gör",
  },
  "workspace.noCandidatesFound": {
    en: "No candidates found.",
    tr: "Aday bulunamadı.",
  },
  "workspace.importCvsToStart": {
    en: "Import CVs from the Setup pane to get started.",
    tr: "Başlamak için Kurulum panelinden CV'leri içe aktarın.",
  },
  "workspace.pageOf": {
    en: "Page {page} of {totalPages} ({total} candidates)",
    tr: "Sayfa {page} / {totalPages} ({total} aday)",
  },
  "workspace.previous": {
    en: "Previous",
    tr: "Önceki",
  },
  "workspace.next": {
    en: "Next",
    tr: "Sonraki",
  },
  "workspace.exportJob": {
    en: "Export Job",
    tr: "İşi Dışa Aktar",
  },
  "workspace.chooseExport": {
    en: "Choose what to export and where to save it.",
    tr: "Ne dışa aktarılacağını ve nereye kaydedileceğini seçin.",
  },
  "workspace.resultsCsv": {
    en: "Results CSV",
    tr: "Sonuçlar CSV",
  },
  "workspace.jobBundle": {
    en: "Job Bundle",
    tr: "İş Paketi",
  },
  "workspace.bundleOptions": {
    en: "Bundle Options",
    tr: "Paket Seçenekleri",
  },
  "workspace.includeJobAd": {
    en: "Include Job Ad",
    tr: "İş İlanını Dahil Et",
  },
  "workspace.includeWeights": {
    en: "Include Weights",
    tr: "Ağırlıkları Dahil Et",
  },
  "workspace.includeResults": {
    en: "Include Results",
    tr: "Sonuçları Dahil Et",
  },
  "workspace.includeLikes": {
    en: "Include Likes",
    tr: "Beğenileri Dahil Et",
  },
  "workspace.exportCsv": {
    en: "Export CSV",
    tr: "CSV Dışa Aktar",
  },
  "workspace.exportBundle": {
    en: "Export Bundle",
    tr: "Paket Dışa Aktar",
  },
  "workspace.exportSuccess": {
    en: "exported successfully!",
    tr: "başarıyla dışa aktarıldı!",
  },
  "workspace.exportError": {
    en: "Failed to export",
    tr: "Dışa aktarma başarısız",
  },
  "workspace.cancel": {
    en: "Cancel",
    tr: "İptal",
  },
  "workspace.save": {
    en: "Save",
    tr: "Kaydet",
  },
  "workspace.rubricWeights": {
    en: "Rubric Weights",
    tr: "Değerlendirme Ölçeği Ağırlıkları",
  },
  "workspace.adjustMaxPoints": {
    en: "Adjust max points for each rubric question.",
    tr: "Her değerlendirme sorusu için maksimum puanları ayarlayın.",
  },
  "workspace.jobAdWeight": {
    en: "Job Ad Weight",
    tr: "İş İlanı Ağırlığı",
  },
  "workspace.jobFitWeight": {
    en: "Job Fit Weight",
    tr: "İş Uyumu Ağırlığı",
  },
  "workspace.crossJobFeedbackWeight": {
    en: "Cross-Job Feedback Weight",
    tr: "İşler Arası Geri Bildirim Ağırlığı",
  },
  "workspace.maxPoints": {
    en: "max points",
    tr: "maks. puan",
  },
  "workspace.cvInspector": {
    en: "CV Inspector",
    tr: "CV İnceleyici",
  },
  "workspace.fileType": {
    en: "File Type",
    tr: "Dosya Türü",
  },
  "workspace.extractionStatus": {
    en: "Extraction Status",
    tr: "Çıkarma Durumu",
  },
  "workspace.scores": {
    en: "Scores",
    tr: "Puanlar",
  },
  "workspace.totalPoints": {
    en: "Total Points",
    tr: "Toplam Puan",
  },
  "workspace.maxPointsLabel": {
    en: "Max Points",
    tr: "Maks. Puan",
  },
  "workspace.rubricBreakdown": {
    en: "Rubric Breakdown",
    tr: "Değerlendirme Ölçeği Detayı",
  },
  "workspace.extractedText": {
    en: "Extracted Text",
    tr: "Çıkarılan Metin",
  },
  "workspace.cvText": {
    en: "CV Text",
    tr: "CV Metni",
  },
  "workspace.matchLens": {
    en: "Match Lens",
    tr: "Eşleşme Merceği",
  },
  "workspace.selectTarget": {
    en: 'Select a target and click "Run Match" to analyze.',
    tr: 'Bir hedef seçin ve analiz etmek için "Eşleştirmeyi Çalıştır"a tıklayın.',
  },
  "workspace.runMatch": {
    en: "Run Match",
    tr: "Eşleştirmeyi Çalıştır",
  },
  "workspace.chunkSize": {
    en: "Chunk size",
    tr: "Parça boyutu",
  },
  "workspace.tokens": {
    en: "tokens",
    tr: "token",
  },
  "workspace.model": {
    en: "Model",
    tr: "Model",
  },
  "workspace.chunks": {
    en: "Chunks",
    tr: "Parçalar",
  },
  "workspace.topMatchingChunks": {
    en: "Top Matching Chunks",
    tr: "En İyi Eşleşen Parçalar",
  },
  "workspace.chunk": {
    en: "Chunk",
    tr: "Parça",
  },
  "workspace.comparatorMissingTitle": {
    en: "Comparator model missing",
    tr: "Karşılaştırıcı model eksik",
  },
  "workspace.comparatorMissingDescription": {
    en: "You're having a subpar experience.",
    tr: "Eksik bir deneyim yaşıyorsunuz.",
  },
  "workspace.getComparator": {
    en: "Get Comparator",
    tr: "Karşılaştırıcıyı Al",
  },
  "workspace.loadingJobAd": {
    en: "Loading job ad...",
    tr: "İş ilanı yükleniyor...",
  },
  "workspace.rankStatus": {
    en: "Rank Status",
    tr: "Sıralama Durumu",
  },
  "workspace.rerank": {
    en: "Re-rank",
    tr: "Yeniden Sırala",
  },
  "workspace.failedToLoadJobAd": {
    en: "Failed to load job ad",
    tr: "İş ilanı yüklenemedi",
  },
  "workspace.empty": {
    en: "(empty)",
    tr: "(boş)",
  },
  "workspace.changeAd": {
    en: "Change ad",
    tr: "İlanı değiştir",
  },
  "workspace.openInEditor": {
    en: "Open in editor",
    tr: "Düzenleyicide aç",
  },
  "workspace.reload": {
    en: "Reload",
    tr: "Yenile",
  },
  "workspace.failedToChangeJobAdPath": {
    en: "Failed to change job ad path",
    tr: "İş ilanı yolu değiştirilemedi",
  },
  "workspace.failedToReloadJobAd": {
    en: "Failed to reload job ad",
    tr: "İş ilanı yenilenemedi",
  },
  "workspace.failedToOpenFileInEditor": {
    en: "Failed to open file in editor",
    tr: "Dosya düzenleyicide açılamadı",
  },
  "workspace.rubricLabels.quick": {
    en: "Quick (fast embed)",
    tr: "Hızlı (hızlı gömme)",
  },
  "workspace.rubricLabels.jobFitEmbed": {
    en: "Job fit (embed)",
    tr: "İş uyumu (gömme)",
  },
  "workspace.rubricLabels.jobFitComparator": {
    en: "Job fit (comparator)",
    tr: "İş uyumu (karşılaştırıcı)",
  },
  "workspace.rubricLabels.feedbackThisJob": {
    en: "Feedback (this job)",
    tr: "Geri bildirim (bu iş)",
  },
  "workspace.rubricLabels.feedbackAllJobs": {
    en: "Feedback (all jobs)",
    tr: "Geri bildirim (tüm işler)",
  },
  "settings.title": {
    en: "Settings",
    tr: "Ayarlar",
  },
  "settings.language": {
    en: "Language",
    tr: "Dil",
  },
  "settings.languageDescription": {
    en: "Choose the language used across the application.",
    tr: "Uygulamanın her yerinde kullanılacak dili seçin.",
  },
  "rank.status.running": {
    en: "Running",
    tr: "Çalışıyor",
  },
  "rank.status.stale": {
    en: "Stale",
    tr: "Eski",
  },
  "rank.status.converged": {
    en: "Converged",
    tr: "Yakındı",
  },
  "rank.status.cancelled": {
    en: "Cancelled",
    tr: "İptal edildi",
  },
  "rank.status.partial": {
    en: "Partial",
    tr: "Kısmi",
  },
  "rank.status.clean": {
    en: "Clean",
    tr: "Temiz",
  },
  "rank.reason.no-cvs": {
    en: "No CVs imported",
    tr: "CV içe aktarılmadı",
  },
  "rank.reason.no-jobad": {
    en: "No job ad set",
    tr: "İş ilanı ayarlanmadı",
  },
  "rank.reason.engine-not-ready": {
    en: "Engine not ready",
    tr: "Motor hazır değil",
  },
  "rank.reason.missing-rubric-vectors": {
    en: "Missing rubric vectors",
    tr: "Değerlendirme vektörleri eksik",
  },
  "rank.reason.candidate-embedding-failed": {
    en: "Candidate embedding failed",
    tr: "Aday gömme başarısız",
  },
  "rank.reason.no-scoreable-candidates": {
    en: "No scoreable candidates",
    tr: "Puanlanabilir aday yok",
  },
  "rank.reason.no-active-embedder": {
    en: "No active embedder",
    tr: "Aktif gömme modeli yok",
  },
  "rank.reason.stale": {
    en: "Ranking is stale",
    tr: "Sıralama eski",
  },
  "rank.reason.clean": {
    en: "Ready to rank",
    tr: "Sıralamaya hazır",
  },
} as const
