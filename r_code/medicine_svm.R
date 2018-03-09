require(stats)
require(dplyr)
require(stats)
require(dplyr)
require(stringr)
require(tidyr)
require(tidytext)
require(tibble)
require(SnowballC)
require(reshape2)
require(onehot)

#for svm
require(kernlab)
require(e1071)

#for xgboost 
require(xgboost)
require(DiagrammeR)

#load raw data
setwd("/Users/hodong/Desktop/jupyter_prac/kaggle_medicine/kaggle_medicine/raw_data")
trv <- data.frame(read.csv("training_variants"))
tev <- data.frame(read.csv("test_variants.csv"))

temp <- readLines("training_text")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
trxt <- data_frame(ID=temp[,1], text=temp[,2])

temp <- readLines("test_text.csv")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
text <- data_frame(ID=temp[,1], text=temp[,2])

#load processed data
setwd("/Users/hodong/Desktop/jupyter_prac/kaggle_medicine/kaggle_medicine/source")
tr_feature <- data.frame(read.csv("tr_feature.csv"))
te_feature <- data.frame(read.csv("te_feature.csv"))
class_word <- data.frame(read.csv("class_word.csv"))
class_word_tf <- data.frame(read.csv("class_word_tf.csv"))
class_bigram <- data.frame(read.csv("class_bigram.csv"))
class_bigram_tf <- data.frame(read.csv("class_bigram_tf.csv"))



#data preparation for model
{
  data("stop_words")
  top_word <- function(x, y){ #텍스트 파일에서 y개의 top frequency 단어 추출
    temp <- x %>% 
      unnest_tokens(word, text, to_lower=TRUE) %>%
      mutate(word=wordStem(word)) %>%
      group_by(word) %>%
      count() %>%
      arrange(desc(n)) %>%
      head(n=y) %>%
      select(word, n)
    
    return(temp)
  }
  top_20_word <- top_word(trxt, 20) 
  
  ##word token 생성
  tr_word_token <- trxt %>% 
    unnest_tokens(word, text) %>%
    mutate(word=wordStem(word)) %>%
    count(ID, word) %>% 
    merge(trv, by="ID") %>%
    select(ID, word, n, Class)
  
  tr_word_token <-  tr_word_token %>% 
    filter(!word %in% top_20_word$word) %>%
    filter(!word %in% stop_words$word)
  
  word_filter <- tr_word_token %>%
    bind_tf_idf(word, ID, n) %>%
    select(word, tf_idf) %>%
    unique() %>%
    arrange(tf_idf) %>% 
    select(word) %>%
    unique() %>%
    head(n=30)
  
  tr_word_token <-  tr_word_token %>% 
    filter(!word %in% word_filter$word) 
  #Let's remove top_20_word and stop_words at once.
  
  te_word_token <- text  %>%
    unnest_tokens(word, text) %>%
    mutate(word=wordStem(word)) %>%
    count(ID, word) %>%
    filter(!word %in% top_20_word$word) %>%
    filter(!word %in% stop_words$word) %>% 
    filter(!word %in% word_filter$word) %>%
    merge(tev, by="ID") %>%
    select(ID, word, n)
  
  
  ##bigram token 생성
  tr_bigram_token <- trxt %>% 
    unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
    separate(bigram, c('w1','w2'), sep=" ") %>%
    mutate(w1=wordStem(w1)) %>%
    mutate(w2=wordStem(w2)) %>%
    filter(!w1 %in% stop_words$word) %>%
    filter(!w2 %in% stop_words$word) %>%
    filter(!w1 %in% top_20_word$word) %>%
    filter(!w2 %in% top_20_word$word) %>%
    unite(bigram, w1, w2, sep=" ") %>%
    count(ID, bigram) %>% 
    merge(trv, by="ID") %>%
    select(ID, bigram, n, Class)
  
  bigram_filter <- tr_bigram_token %>%
    bind_tf_idf(bigram, ID, n) %>%
    select(bigram, tf_idf) %>%
    unique() %>%
    arrange(tf_idf) %>% 
    select(bigram) %>%
    unique() %>%
    head(n=15)
  
  tr_bigram_token <- tr_bigram_token %>%
    filter(!bigram %in% bigram_filter$bigram)
  
  te_bigram_token <- text %>%
    unnest_tokens(bigram, text, token="ngrams", n=2) %>%
    separate(bigram, c('w1','w2'), sep=" ") %>%
    mutate(w1=wordStem(w1)) %>%
    mutate(w2=wordStem(w2)) %>%
    filter(!w1 %in% stop_words$word) %>%
    filter(!w2 %in% stop_words$word) %>%
    filter(!w1 %in% top_20_word$word) %>%
    filter(!w2 %in% top_20_word$word) %>%
    unite(bigram, w1, w2, sep=" ") %>%
    filter(!bigram %in% bigram_filter$bigram) %>%
    count(ID, bigram) %>% 
    merge(tev, by="ID") %>%
    select(ID, bigram, n) 
  
  head(class_word)
  head(tr_word_token)
  head(te_word_token)
  head(class_bigram)
  head(tr_bigram_token)
  head(te_bigram_token)
}

##multi-class svm
#multiclass svm with all feature (p>100)
multi_svm <- function(feature=feature, tr_data=tr_data, te_data=te_data, by=by, token=token, params=param){
  
  tr_dcg <- data.frame(sort(as.numeric(unique(tr_data$ID))))
  names(tr_dcg) <- "ID"
  te_dcg <- data.frame(sort(as.numeric(unique(te_data$ID))))
  names(te_dcg) <- "ID"
  
  for(i in c(1:9)){
    trn <- freq_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="test")
    tes <- freq_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
    tr_dcg <- data.frame(cbind(tr_dcg, trn %>% select(-ID)))
    te_dcg <- data.frame(cbind(te_dcg, tes %>% select(-ID)))
  }
  
  tr_dcg <- tr_dcg %>%
    merge(trv, by="ID") %>%
    select(-Gene, -Variation) %>%
    mutate(ID=as.numeric(ID)) %>%
    arrange(ID)
  
  tr_dcg <- tr_dcg[, !duplicated(colnames(tr_dcg))]
  svm_train <- tr_dcg %>% select(-ID) %>% mutate(Class=as.factor(Class))
  trn_id <- tr_dcg %>% select(ID)
  
  svm_model <- ksvm(Class~., data=svm_train, kernel="rbfdot", prob.model=TRUE)
  svm_temp <- predict(svm_model, te_dcg[,-1], type="probabilities")
  
  svm_result <- data.frame(cbind(te_dcg$ID, svm_temp))
  names(svm_result) <- c("ID", c(1:9))
  
  result_table(svm_result, te_word_label)
  
  return(svm_result)  
}
svm_table <- function(feature=x, data=y, by=z, token=w, i=i, pur=pur){
  #feature : class_word처럼 분류할 label별 word 혹은 bigram 등
  #data : tr_word_token처럼 document(ID)별 tokenized된 word와 bigram 목록과 label
  #by : frequency 기준일지, tf-idf를 이용할 것인지
  #token : word를 이용할 것인지, bigram을 이용할 것인지
  
  if(pur=="train"){
    cl_info <- data %>%
      select(ID, Class) %>%
      distinct()
  }
  
  feature <- feature %>% #i번째 label에 해당하는 feature set만 유지
    filter(Class==i)
  
  if(token=="word"){
    if(by=="tf_idf"){
      feature <- feature %>%
        mutate(n=tf_idf) %>%
        select(-tf_idf)
      
      data <- data %>%
        bind_tf_idf(word, ID, n) %>%
        select(-n, -tf, -idf) %>%
        mutate(n=tf_idf) %>%
        select(-tf_idf)
      data[is.na(data$n),] <- 0
    }
    
    crs_join <- merge(unique(data %>% select(ID)), feature$word, by=NULL) %>%
      mutate(word=y) %>%
      select(-y) %>%
      arrange(as.numeric(ID))
    
    ft_vec <- as.character(unique(feature$word))
    
    data <- data %>%
      filter(word %in% ft_vec)
    
    lft_join <- merge(crs_join, data, all.x="TRUE") %>%
      arrange(as.numeric(ID))
    lft_join[is.na(lft_join)] <- 0
    
    tab <- dcast(lft_join, ID~word, value.var="n", fill=0) %>%
      arrange(as.numeric(ID))
  } 
  
  if(token=="bigram"){
    if(by=="tf_idf"){
      feature <- feature %>%
        mutate(n=tf_idf) %>%
        select(-tf_idf)
      
      data <- data %>%
        bind_tf_idf(bigram, ID, n) %>%
        select(-n, -tf, -idf) %>%
        mutate(n=tf_idf) %>%
        select(-tf_idf)
      data[is.na(data$n),] <- 0
    }
    
    crs_join <- merge(unique(data %>% select(ID)), feature$bigram, by=NULL) %>%
      mutate(bigram=y) %>%
      select(-y) %>%
      arrange(as.numeric(ID))
    
    ft_vec <- as.character(t(feature$bigram))
    
    data <- data %>%
      filter(bigram %in% ft_vec)
    
    lft_join <- merge(crs_join, data, all.x="TRUE") %>%
      arrange(as.numeric(ID))
    lft_join[is.na(lft_join)] <- 0
    
    tab <- dcast(lft_join, ID~bigram, value.var="n", fill=0) %>%
      arrange(as.numeric(ID))
  }
  
  if(pur=="train"){
    tab <- tab %>%
      merge(cl_info, by="ID")
    tab[which(tab$Class!=i),]$Class <- 0
    tab <- tab %>% 
      mutate(Class=as.factor(Class))
  }
  
  tab <- tab %>%
    mutate(ID=as.numeric(ID)) %>%
    arrange(ID)
  
  if(pur=="train"){
    v <- c()
    for(j in c(2:ncol(tab)-1)){
      if(sum(as.numeric(tab[,j]))==0){ v <- c(v,j)}
    }
    if(length(v)!=0){ tab <- tab[,-v] }
  }
  
  return(tab)
}
#multiclass svm made of binary svm classifiers and softmax
multi_class_svm <- function(feature=feature, tr_data=tr_data, te_data=te_data, by=by, token=token, kern=kern){
  tr_len <- trxt %>%
    mutate(text_len=log10(str_length(text))) %>%
    select(ID, text_len)
  
  te_len <- text %>%
    mutate(text_len=log10(str_length(text))) %>%
    select(ID, text_len)
  
  tr_gene <- onehot_gene(trv, 30)
  te_gene <- onehot_gene(tev, 30)
  
  result <- data.frame(matrix(nrow=length(unique(te_data$ID)),ncol=0))
  test_id <- sort(as.numeric(unique(te_data$ID)))
  for(i in c(1:9)){
    svm_train <- svm_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="train")
    svm_test <- svm_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
    svm_test <- svm_test[,(names(svm_test) %in% names(svm_train))]
    
    svm_train <- svm_train %>% 
      merge(data.frame(cbind(tr_len, tr_gene)), by="ID")
    
    svm_test <- svm_test %>% 
      merge(data.frame(cbind(tr_len, tr_gene)), by="ID")
    
    svm_model <- ksvm(Class~., data=svm_train[,-1], kernel=kern, prob.model=TRUE)
    svm_temp <- predict(svm_model, svm_test[,-1], type="probabilities")
    
    result <- data.frame(cbind(result, svm_temp[,2]))
  }
  result <- t(apply(result, 1, softmax))
  result <- data.frame(cbind(test_id, result))
  names(result) <- c("ID", as.character(c(1:9)))
  
  return(result)
}
#svm_result <- multi_class_svm(feature=class_word, tr_data=tr_word_token, te_data=te_word_token, by="n", token="word", kern="rbfdot")

#training set을 tr과 valid set으로 나눠서 작성해본다. 
set.seed(171213)
sam_num <- sample(nrow(trv), 2200)
ID_list <- sort(unique(trv$ID))
tr_num <- ID_list[sam_num]
te_num <- ID_list[-sam_num]

tr_word_data <- tr_word_token %>%
  filter(ID %in% tr_num)

te_word_data <- tr_word_token %>%
  filter(ID %in% te_num) %>%
  select(-Class)

te_word_label <- tr_word_token %>%
  filter(ID %in% te_num) %>%
  select(ID, Class) %>%
  mutate(ID=as.numeric(ID)) %>%
  arrange(ID) %>%
  distinct()

tr_bigram_data <- tr_bigram_token %>%
  filter(ID %in% tr_num)

te_bigram_data <- tr_bigram_token %>%
  filter(ID %in% te_num) %>%
  select(-Class)

te_bigram_label <- tr_bigram_token %>%
  filter(ID %in% te_num) %>%
  select(ID, Class) %>%
  mutate(ID=as.numeric(ID)) %>%
  arrange(ID) %>%
  distinct()

result_compare <- data.frame(cbind(c(1:9),matrix(nrow=9,ncol=0)))


#----------------#support_vector_machine model#----------------#
#model using word & n
svm_result <- multi_class_svm(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", kern="rbfdot")
temp <- result_table(svm_result, te_word_label) #accuracy : 0.5628
result_compare <- cbind(result_compare, temp[,-1])

#model using word & tf_idf
svm_result <- multi_class_svm(feature=class_word_tf, tr_data=tr_word_data, te_data=te_word_data, by="tf_idf", token="word", kern="rbfdot")
temp <- result_table(svm_result, te_word_label) #accuracy : 0.4316
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & n
svm_result <- multi_class_svm(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", kern="rbfdot")
temp <- result_table(svm_result, te_bigram_label) #accuracy : 0.5361
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & tf_idf
svm_result <- multi_class_svm(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", kern="rbfdot")
temp <- result_table(svm_result, te_bigram_label) #accuracy : 0.4549
result_compare <- cbind(result_compare, temp[,-1])


