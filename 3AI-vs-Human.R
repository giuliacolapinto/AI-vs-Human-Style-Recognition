# 3 AI vs HUMAN -----------------------------------------------------------

rm(list = ls())
library(tidyverse)
library(NLP)
library(tm)
library(quanteda)
library(readr)

dataset <- read_csv("set/your/path/train.csv")
dataset <- na.omit(dataset)

GPT <- dataset %>%
  dplyr::select(`GPT_4-o`)%>%
  rename(testo = `GPT_4-o`)
GPT$label <- as.factor(rep(1,nrow(GPT)))

human <- dataset %>%
  dplyr::select(Human_story)%>%
  rename(testo = Human_story)
human$label <- as.factor(rep(0,nrow(human)))

llama <- dataset %>%
  dplyr::select(`llama-8B`)%>%
  rename(testo = `llama-8B`)
llama$label <- as.factor(rep(1,nrow(llama)))

gemma <- dataset %>%
  dplyr::select(`gemma-2-9b`)%>%
  rename(testo = `gemma-2-9b`)
gemma$label <- as.factor(rep(1,nrow(gemma)))


data <- rbind(human, GPT[1:2418,])
data <- rbind(data, llama[2419:4836,])
data <- rbind(data, gemma[4837:7255,])
data$label

n <- nrow(data)
set.seed(123)
sample <- sample(1:n, size=30/100*n)

training <- data[-sample,]
test <- data[sample,]

sum(training$label==1)/length(training$label)
sum(test$label==1)/length(test$label)

corpus <- Corpus(VectorSource(training$testo))
corpus

corpus <- tm_map(corpus,stemDocument)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus,content_transformer(tolower))

tokens <- tokens(corpus(corpus), 
                 remove_punct = T,
                 remove_numbers = T )
ngrams <- tokens_ngrams(tokens, n = 1:2)
ngrams_trim <- tokens_trim(ngrams, min_termfreq=1000)

dfm <- dfm(ngrams_trim)
dfm_tf_idf <- dfm_tfidf(dfm)

library(tokenizers)

features <- data.frame(testo = training$testo) %>%
  mutate(
    n_parole = count_words(testo),
    n_frasi = count_sentences(testo),
    l_frase = n_parole / n_frasi,
    punteggiatura = str_count(testo,"[.,;:!?]"),
    stopwords = ntoken(tokens_select(tokens,
                                     pattern = stopwords("en"))),
    uniche = rowSums(dfm == 1)
  )

# Feature Exploration

feat_plot <- features %>% mutate(label = as.factor(training$label))

cat("\nDESCRIPTIVE STATISTICS OF FEATURES\n")
cat("\n Human = 0 | Mix 3 AI = 1 \n")
desc_stats <- feat_plot %>%
  group_by(label) %>%
  summarise(
    Mean_Parole = mean(n_parole),
    Median_Parole = median(n_parole),
    Mean_Frasi = mean(n_frasi),
    Mean_Len_Frase = mean(l_frase),
    Mean_Stopwords_Ratio = mean(stopwords),
    Mean_Uniche = mean(uniche)
  )
print(desc_stats)

my_colors <- c("#8A2BE2", "#00FF7F") 

p_len <- ggplot(feat_plot, aes(x = label, y = l_frase, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Mean Sentence Length Comparison", x = "Class (0 = Human, 1 = AI)", y = "Words per Sentence") +
  theme_minimal() + theme(legend.position = "none")

p_stop <- ggplot(feat_plot, aes(x = label, y = stopwords, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Stopword Usage Comparison", x = "Class (0 = Human, 1 = AI)", y = "Stopwords / Total Words") +
  theme_minimal() + theme(legend.position = "none")

p_n_frasi <- ggplot(feat_plot, aes(x = label, y = n_frasi, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Number of Sentences Comparison", x = "0 = Human, 1 = AI", y = "N. Sentences") +
  theme_minimal() + theme(legend.position = "none")

p_n_parole <- ggplot(feat_plot, aes(x = label, y = n_parole, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Number of Words Comparison", x = "0 = Human, 1 = AI", y = "N. Words") +
  theme_minimal() + theme(legend.position = "none")

p_uniche <- ggplot(feat_plot, aes(x = label, y = uniche, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Unique Words Comparison", x = "0 = Human, 1 = AI", y = "Amount of Unique Words") +
  theme_minimal() + theme(legend.position = "none")

X_temp <- as.matrix(dfm_tf_idf)
get_word_freq <- function(matrice, word) {
  if(word %in% colnames(matrice)) return(matrice[, word])
  return(rep(0, nrow(matrice)))
}

df_words <- data.frame(
  label = as.factor(training$label),
  site_search = get_word_freq(X_temp, "site_search"),
  lee = get_word_freq(X_temp, "lee"),
  becaus = get_word_freq(X_temp, "becaus")
)

p_site <- ggplot(df_words, aes(x = label, y = site_search, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Comparison of the word 'site_search'", x = "0 = Human, 1 = AI", y = "TF-IDF") +
  theme_minimal() + theme(legend.position = "none")

p_becaus <- ggplot(df_words, aes(x = label, y = becaus, fill = label)) + 
  geom_boxplot(alpha = 0.8) + 
  scale_fill_manual(values = my_colors) +
  labs(title = "Comparison of the word 'becaus'", x = "0 = Human, 1 = AI", y = "TF-IDF") +
  theme_minimal() + theme(legend.position = "none")

print(p_len)
print(p_stop)
print(p_n_frasi)
print(p_n_parole)
print(p_uniche)
print(p_site)
print(p_becaus)

features <- as.matrix(scale(features[,-1]))

X <- as.matrix(dfm_tf_idf)
X <- cbind(X,features)

str(X)
X <- as.matrix(X)
sum(apply(X, 2, var) == 0)

dim(X)

pc <- prcomp(X, scale. = T, center = T)

varianza_spiegata <- cumsum(pc$sdev^2 / sum(pc$sdev^2))
k <- which(varianza_spiegata > 0.7)[1]

# PCA Interpretation
df_var <- data.frame(Component = 1:length(varianza_spiegata), Variance = varianza_spiegata)
p_var <- ggplot(df_var, aes(x = Component, y = Variance)) +
  geom_line(color="#8A2BE2", linewidth=1) +
  geom_hline(yintercept = 0.7, linetype="dashed", color = "#00FF7F") +
  geom_vline(xintercept = k, linetype="dashed", color = "#00FF7F") +
  labs(title = paste("Scree Plot: Explained Variance (70% at k =", k, ")"),
       x = "Number of Principal Components", y = "Cumulative Variance") +
  theme_minimal()
print(p_var)

X_pca <- pc$x[, 1:k]

X_pca_labels <- cbind(X_pca, training$label)
X_pca_labels <- as.data.frame(X_pca_labels) %>%
  rename(label = V107)

p_pca <- ggplot(X_pca_labels, aes(x = PC1, y = PC2, color = as.factor(label))) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = my_colors) +
  labs(title = "PCA Projection: PC1 vs PC2", subtitle = "0 = Human | 1 = Mix 3 AI", x = "PC1", y = "PC2", color = "Class") +
  theme_minimal()
print(p_pca)

# Training set ------------------------------------------------------------

library(MASS)
lda = lda(label ~ ., data = X_pca_labels)

qda = qda(label ~ ., data = X_pca_labels)

m1 <- glm(as.factor(label) ~ ., data = X_pca_labels, family=binomial)

# Test set ----------------------------------------------------------------

corpus_test <- Corpus(VectorSource(test$testo))
corpus_test


corpus_test <- tm_map(corpus_test,stemDocument)
corpus_test <- tm_map(corpus_test, stripWhitespace)
corpus_test <- tm_map(corpus_test,content_transformer(tolower))

tokens_test <- tokens(corpus(corpus_test), 
                      remove_punct = T,
                      remove_numbers = T )
ngrams_test <- tokens_ngrams(tokens_test, n = 1:2)

dfm_test <- dfm(ngrams_test)
dfm_test <- dfm_match(dfm_test, features = featnames(dfm))
dfm_tf_idf_test <- dfm_tfidf(dfm_test)


features_test <- data.frame(testo = test$testo) %>%
  mutate(
    n_parole = count_words(testo),
    n_frasi = count_sentences(testo),
    l_frase = n_parole / n_frasi,
    punteggiatura = str_count(testo,"[.,;:!?]"),
    stopwords = ntoken(tokens_select(tokens_test,
                                     pattern = stopwords("en"))),
    uniche = rowSums(dfm_test == 1)
  )
features_test <- as.matrix(scale(features_test[,-1]))

X_test <- as.matrix(dfm_tf_idf_test)
X_test <- cbind(X_test,features_test)

X_test_pca <- predict(pc, X_test)[, 1:k]

# LDA ----------------------------------------------------------

pred <- predict(lda, as.data.frame(X_test_pca))
levels(pred$class) <- c(0,1)

library(caret)
cf <- confusionMatrix(pred$class, test$label)
cf   
fourfoldplot(cf$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: LDA")
# Accuracy: 0.9681

# QDA ----------------------------------------------------------

pred_qda = predict(qda, as.data.frame(X_test_pca))
levels(pred_qda$class) <- c(0,1)


cf <- confusionMatrix(pred_qda$class, test$label)
cf   
fourfoldplot(cf$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: QDA")
# Accuracy: 0.8075

# LOGISTIC REGRESSION ---------------------------------------------------------------

pred_logi <- predict(m1, as.data.frame(X_test_pca), type="response")
pred_class <- ifelse(pred_logi  > 0.5, "1", "0")
pred_class <- as.factor(pred_class)

cf_log <- confusionMatrix(pred_class, test$label)
print(cf_log)
fourfoldplot(cf_log$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: Logistic Regression")
# Accuracy: 0.9812

# KNN ---------------------------------------------------------------------

library(class)

# Select best K manually (up to 15)
k_values <- 1:15
accuracies <- numeric(length(k_values))
for (i in seq_along(k_values)) {
  pred_k <- knn(train = X_pca, test = X_test_pca, cl = training$label, k = k_values[i])
  accuracies[i] <- mean(pred_k == test$label)
}
best_k <- k_values[which.max(accuracies)]

cat("Best K found:", best_k, "\n")
plot(k_values, accuracies, type = "b", col = "#8A2BE2", pch = 19,
     main = "KNN: Accuracy vs K", xlab = "K", ylab = "Accuracy")
abline(v = best_k, col = "#00FF7F", lty = 2)

# Fit model with best_k
pred <- knn(train = X_pca, test = X_test_pca, cl = training$label, k = best_k)

cf_knn <- confusionMatrix(pred, test$label)
print(cf_knn)
fourfoldplot(cf_knn$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: KNN")
# Accuracy: ~0.9674

# Decision Tree -----------------------------------------------------------

library(tree)

train_tree <- data.frame(X, label = training$label)
test_tree  <- data.frame(X_test)

tree_model <- tree(label ~ ., data = train_tree)
cv <- cv.tree(tree_model , FUN = prune.misclass)

plot(cv$size , cv$dev, type = "b")
plot(cv$k, cv$dev, type = "b")

prune <- prune.misclass(tree_model , best = 9)
plot(prune)
text(prune, pretty = 0)

pred_tree <- predict(prune, test_tree, type = "class")
mean(pred_tree == test$label)
# Accuracy: 0.9191

# External Test Set -------------------------------------------------

testt <- read_csv("set/your/path/Training_Essay_Data.csv")
testt <- na.omit(testt[1:5000,]) %>%
  rename(testo=text,
         label=generated) 

testt$label <- as.factor(testt$label) 

corpus_testt <- Corpus(VectorSource(testt$testo))

corpus_testt <- tm_map(corpus_testt, stemDocument)
corpus_testt <- tm_map(corpus_testt, stripWhitespace)
corpus_testt <- tm_map(corpus_testt, content_transformer(tolower))

tokens_testt <- tokens(corpus(corpus_testt), 
                      remove_punct = T,
                      remove_numbers = T,
                      remove_symbols=T)
ngrams_testt <- tokens_ngrams(tokens_testt, n = 1:2)

dfm_testt <- dfm(ngrams_testt)
dfm_testt <- dfm_match(dfm_testt, features = featnames(dfm))
dfm_tf_idf_testt <- dfm_tfidf(dfm_testt)

features_testt <- data.frame(testo = testt$testo) %>%
  mutate(
    n_parole = count_words(testo),
    n_frasi = count_sentences(testo),
    l_frase = n_parole / n_frasi,
    punteggiatura = str_count(testo,"[.,;:!?]"),
    stopwords = ntoken(tokens_select(tokens_testt,
                                     pattern = stopwords("en"))),
    uniche = rowSums(dfm_testt == 1)
  )
features_testt <- as.matrix(scale(features_testt[,-1]))

X_testt <- as.matrix(dfm_tf_idf_testt)
X_testt <- cbind(X_testt, features_testt)

X_testt_pca <- predict(pc, X_testt)[, 1:k]

# LDA ---------------------------------------------------------------------

pred_testt <- predict(lda, as.data.frame(X_testt_pca))
levels(pred_testt$class) <- c(0,1)

cf_lda_testt <- confusionMatrix(pred_testt$class, testt$label)
print(cf_lda_testt)   
fourfoldplot(cf_lda_testt$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: LDA (External)")
# Accuracy: 0.913

# QDA ---------------------------------------------------------------------

pred_qda_testt = predict(qda, as.data.frame(X_testt_pca))
levels(pred_qda_testt$class) <- c(0,1)

cf_qda_testt <- confusionMatrix(pred_qda_testt$class, testt$label)
print(cf_qda_testt)   
fourfoldplot(cf_qda_testt$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: QDA (External)")
# Accuracy: 0.3152

# LOGISTIC REGRESSION ---------------------------------------------------------------

pred_logi_testt <- predict(m1, as.data.frame(X_testt_pca), type="response")
pred_class_testt <- ifelse(pred_logi_testt  > 0.5, "1", "0")
pred_class_testt <- as.factor(pred_class_testt)

cf_log_testt <- confusionMatrix(pred_class_testt, testt$label)
print(cf_log_testt)
fourfoldplot(cf_log_testt$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: Logistic Regression (External)")
# Accuracy: ~0.9232 (with Best K)

# KNN ---------------------------------------------------------------------

pred_knn_testt <- knn(train = X_pca, test = X_testt_pca, cl = training$label, k = best_k)

cf_knn_testt <- confusionMatrix(pred_knn_testt, testt$label)
print(cf_knn_testt)
fourfoldplot(cf_knn_testt$table, color = c("#00FF7F", "#8A2BE2"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix: KNN (External)")
# Accuracy: ~0.9232 (with Best K)

# Decision tree -----------------------------------------------------------

test_tree_testt  <- data.frame(X_testt)

pred_tree_testt <- predict(prune, test_tree_testt, type = "class")
mean(pred_tree_testt == testt$label)
# Accuracy: 0.5532

# Papers ------------------------------------------------------------------

papers <- read.csv("set/your/path/arXiv_scientific_dataset.csv", stringsAsFactors = F)

summary(papers)
papers$published_date <- as.Date(papers$published_date, "%m/%d/%y")
papers_1 <- papers %>%
  dplyr::filter(published_date > as.Date("2007-01-01") & published_date < as.Date("2015-12-31")) %>%
  dplyr::filter(summary_word_count > 200) %>%
  dplyr::select(published_date, summary)

papers_2 <- papers %>%
  dplyr::filter(published_date > as.Date("2022-01-01")) %>%
  dplyr::filter(summary_word_count > 200) %>%
  dplyr::select(published_date, summary)

papers_test <- rbind(papers_1, papers_2)

corpus_papers <- Corpus(VectorSource(papers_test$summary))
corpus_papers


corpus_papers <- tm_map(corpus_papers, stemDocument)
corpus_papers <- tm_map(corpus_papers, stripWhitespace)
corpus_papers <- tm_map(corpus_papers, content_transformer(tolower))

tokens_papers <- tokens(corpus(corpus_papers),
  remove_punct = T,
  remove_numbers = T
)
ngrams_papers <- tokens_ngrams(tokens_papers, n = 1:2)

dfm_papers <- dfm(ngrams_papers)

dfm_papers <- dfm_match(dfm_papers, features = featnames(dfm))
dfm_tf_idf_papers <- dfm_tfidf(dfm_papers)

features_papers <- data.frame(testo = papers_test$summary) %>%
  mutate(
    n_parole = count_words(testo),
    n_frasi = count_sentences(testo),
    l_frase = n_parole / n_frasi,
    punteggiatura = str_count(testo, "[.,;:!?]"),
    stopwords = ntoken(tokens_select(tokens_papers,
      pattern = stopwords("en")
    )),
    uniche = rowSums(dfm_papers == 1)
  )

features_papers <- as.matrix(scale(features_papers[, -1]))

X_papers <- as.matrix(dfm_tf_idf_papers)
X_papers <- cbind(X_papers, features_papers)

X_papers_pca <- predict(pc, X_papers)[, 1:k]

# LDA ---------------------------------------------------------------------

pred_papers <- predict(lda, as.data.frame(X_papers_pca))
levels(pred_papers$class) <- c(0, 1)
sum(pred_papers$class == 1)
plot(papers_test$published_date, pred_papers$class)
mean(pred_papers$class[1:nrow(papers_1)] == 1)
mean(pred_papers$class[-(1:nrow(papers_1))] == 1)

# LOGISTIC REGRESSION ---------------------------------------------------------------

pred_logi_papers <- predict(m1, as.data.frame(X_papers_pca), type = "response")
pred_class_papers <- ifelse(pred_logi_papers > 0.5, "1", "0")
pred_class_papers <- as.factor(pred_class_papers)

# KNN ---------------------------------------------------------------------

library(class)
pred_knn_papers <- knn(train = X_pca, test = X_papers_pca, cl = training$label, k = best_k)

sum(pred_knn_papers == 1)
plot(papers_test$published_date, pred_knn_papers, main = "KNN Predictions on Papers")
mean(pred_knn_papers[1:nrow(papers_1)] == 1)
mean(pred_knn_papers[-(1:nrow(papers_1))] == 1)

# Plots

tot_papers_anno <- papers %>%
  mutate(Anno = format(published_date, "%Y")) %>%
  group_by(Anno) %>%
  summarise(Tot_Papers = n(), .groups = "drop")

tot_papers_periodo <- papers %>%
  mutate(Periodo = ifelse(published_date > as.Date("2007-01-01") & published_date < as.Date("2015-12-31"), "Period 1 (2007-2015)", 
                   ifelse(published_date > as.Date("2022-01-01"), "Period 2 (2022+)", "Other"))) %>%
  filter(Periodo != "Other") %>%
  group_by(Periodo) %>%
  summarise(Tot_Papers = n(), .groups = "drop")

tot_papers_mensile <- papers %>%
  filter(published_date >= as.Date("2022-01-01")) %>%
  mutate(Mese_Anno = format(published_date, "%Y-%m")) %>%
  group_by(Mese_Anno) %>%
  summarise(Tot_Papers = n(), .groups = "drop")

results_df <- data.frame(
  Data = papers_test$published_date,
  Class = as.numeric(as.character(pred_class_papers)) # 1 = AI-Like, 0 = Human
) %>%
  mutate(
    Anno = format(Data, "%Y"),
    Periodo = ifelse(Data < as.Date("2016-01-01"), "Period 1 (2007-2015)", "Period 2 (2022+)")
  )

# Comparison of the two periods

p_period <- results_df %>%
  group_by(Periodo) %>%
  summarise(AI_Count = sum(Class == 1), .groups = "drop") %>%
  left_join(tot_papers_periodo, by = "Periodo") %>%
  mutate(AI_Percentage = (AI_Count / Tot_Papers) * 100) %>%
  ggplot(aes(x = Periodo, y = AI_Percentage, fill = Periodo)) +
  geom_bar(stat = "identity", alpha = 0.8, width = 0.5) +
  scale_fill_manual(values = c("#8A2BE2", "#00FF7F")) +
  geom_text(aes(label = sprintf("%.2f%%", AI_Percentage)), vjust = -0.5, fontface = "bold", size = 5) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "Percentage of Papers Classified as AI",
       subtitle = "Standardized by the total papers in the entire dataset",
       y = "AI Percentage (%)",
       x = "") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14))

p_period

cat("\nStatistics on AI Percentage in the two periods (Standardized):\n")
# Same as the first graph but in table format for rmarkdown output

stats_stampa <- results_df %>%
  group_by(Periodo) %>%
  summarise(Documenti_AI = sum(Class == 1),
            .groups = "drop") %>%
  left_join(tot_papers_periodo, by = "Periodo") %>%
  mutate(Percentuale_AI = (Documenti_AI / Tot_Papers) * 100) %>%
  rename(Totale_Papers = Tot_Papers)
print(stats_stampa)

# Annual comparison

p_anno <- results_df %>%
  group_by(Anno) %>%
  summarise(AI_Count = sum(Class == 1), .groups = "drop") %>%
  left_join(tot_papers_anno, by = "Anno") %>%
  mutate(AI_Percentage = (AI_Count / Tot_Papers) * 100) %>%
  ggplot(aes(x = Anno, y = AI_Percentage, group = 1)) +
  geom_bar(stat = "identity", fill = "#8A2BE2", alpha = 0.5) +
  geom_line(color = "#00FF7F", linewidth = 1.2) +
  geom_point(color = "#00FF7F", size = 3) +
  geom_text(aes(label = sprintf("%.2f%%", AI_Percentage)), vjust = -1, size = 3.5, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "AI Trend Evolution",
       subtitle = "% Share relative to all papers available in the dataset for the year",
       y = "AI-Like Percentage (%)",
       x = "Publication Year") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        plot.title = element_text(face = "bold", size = 14))

print(p_anno)

# Total Volume

p_volumi <- results_df %>%
  group_by(Anno) %>%
  summarise(AI_Count = sum(Class == 1), .groups = "drop") %>%
  left_join(tot_papers_anno, by = "Anno") %>%
  mutate(`Others (Human / Unanalyzed)` = Tot_Papers - AI_Count) %>%
  rename(`AI-Like` = AI_Count) %>%
  tidyr::pivot_longer(cols = c(`AI-Like`, `Others (Human / Unanalyzed)`), names_to = "Categoria", values_to = "Conteggio") %>%
  ggplot(aes(x = Anno, y = Conteggio, fill = Categoria)) +
  geom_bar(stat = "identity", position = "stack", alpha = 0.85) +
  scale_fill_manual(values = c("AI-Like" = "#00FF7F", "Others (Human / Unanalyzed)" = "#8A2BE2")) +
  geom_text(aes(label = Conteggio), position = position_stack(vjust = 0.5), size = 3, color = "white", fontface = "bold", check_overlap = TRUE) +
  labs(title = "Absolute Publication Volumes",
       subtitle = "Comparison between AI papers and the total documented volume",
       fill = "Writing Style",
       y = "Number of Papers",
       x = "Year") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "bottom")

p_volumi

# AI-Like Trend in Period 2 (From 2022 onwards)

p_mensile <- results_df %>%
  filter(Data >= as.Date("2022-01-01")) %>%
  mutate(Mese_Anno = format(Data, "%Y-%m")) %>%
  group_by(Mese_Anno) %>%
  summarise(AI_Count = sum(Class == 1), .groups = "drop") %>%
  left_join(tot_papers_mensile, by = "Mese_Anno") %>%
  mutate(AI_Percentage = (AI_Count / Tot_Papers) * 100) %>%
  ggplot(aes(x = Mese_Anno, y = AI_Percentage, group = 1)) +
  geom_area(fill = "#00FF7F", alpha = 0.3) +
  geom_line(color = "#00FF7F", linewidth = 1.2) +
  geom_point(color = "#00FF7F", size = 2) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "Monthly Diffusion Post-2022",
       subtitle = "% Standardization on monthly total",
       y = "AI Percentage (%)",
       x = "Month and Year") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 8, vjust = 0.5), # Vertical to better read the months
        plot.title = element_text(face = "bold", size = 14))

p_mensile