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


#주어진 변수를 이용해 각 ID별 i번째 클래스의 feature를 가진 frequency table로 만듬
freq_table <- function(feature=x, data=y, by=z, token=w, i=i, pur=k){
  #feature : class_word처럼 분류할 label별 word 혹은 bigram 등
  #data : tr_word_token처럼 document(ID)별 tokenized된 word와 bigram 목록과 label
  #by : frequency 기준일지, tf-idf를 이용할 것인지
  #token : word를 이용할 것인지, bigram을 이용할 것인지
  
  feature <- feature %>% #i번째 label에 해당하는 feature set만 유지
    filter(Class==i)
  
  if(pur=="train"){
    data <- data %>% 
      filter(Class==i) #i번째 label에 해당하는 행만 유지
  }
  
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
    }
    
    crs_join <- merge(unique(data %>% select(ID)), feature$word, by=NULL) %>%
      mutate(word=y) %>%
      select(-y) %>%
      arrange(as.numeric(ID))
    
    ft_vec <- as.character(unique(feature$word))
    
    data <- data %>%
      filter(word %in% ft_vec)
    
    if(pur=="train"){
      data <- data %>%
        select(-Class) }
    
    lft_join <- merge(crs_join, data, all.x="TRUE") %>%
      arrange(as.numeric(ID))
    lft_join[is.na(lft_join)] <- 0
    lft_join <- lft_join %>% unique() 
    
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
    }
    
    crs_join <- merge(unique(data %>% select(ID)), feature$bigram, by=NULL) %>%
      mutate(bigram=y) %>%
      select(-y) %>%
      arrange(as.numeric(ID))
    
    ft_vec <- as.character(t(feature$bigram))
    
    data <- data %>%
      filter(bigram %in% ft_vec)
    
    if(pur=="train"){
      data <- data %>%
        select(-Class) }
    
    lft_join <- merge(crs_join, data, all.x="TRUE") %>%
      arrange(as.numeric(ID))
    lft_join[is.na(lft_join)] <- 0
    
    tab <- dcast(lft_join, ID~bigram, value.var="n", fill=0) %>%
      arrange(as.numeric(ID))
  }
  
  return(tab)
}
#주어진 frequency table을 이용해 i번째 클래스의 feature별 관측 probability table로 만듬
prob_mat <- function(freq_tab=x){
  den <- freq_tab %>%
    select(-ID) %>%
    sum()
  
  num <- freq_tab %>%
    select(-ID) %>%
    apply(2, sum)
  aa <- (num+1)/(den+length(num))
  return(matrix(aa))
}
softmax <- function(x){
  return(exp(x+max(x))/sum(exp(x+max(x))))
}
onehot_gene <- function(x, n){
  aa <- trv %>%
    count(Gene) %>%
    arrange(desc(n)) %>%
    top_n(30, n)
  
  qq <- data.frame(cbind(with(x, model.matrix(~Gene + 0))))
  names(qq) <- sub("Gene", "", names(qq))
  qq <- qq %>%
    select(names(qq)[names(qq) %in% intersect(names(qq), aa$Gene)])
  
  return(qq)
}

#freq_table, prob_mat과 주어진 변수를 이용해 naive_baysean 모델을 만듬
#2진으로 계산하면, 앞서 봤던 단어셋들의 분포가 무용해진다. 2진이었던 걸 본래대로 계산하되,
#본래 값인 상태에서 가로로 softmax를 만든다. 
naive_baysean <- function(feature=class_word, data=tr_word_token, by="n", token="word"){
  
  model <- data.frame(matrix(nrow=20, ncol=0))
  for(i in c(1:9)){
    train_table <- freq_table(feature=feature, data=data, by=by, token=token, i=i, pur="train")
    temp <- prob_mat(train_table)
    model <- data.frame(cbind(model, temp))
  }
  names(model) <- as.character(c(1:9))
  
  return(model)  
}
nb_predict <- function(nb_model, feature=class_word, data=te_word_token, by="n", token="word"){
  result <- data.frame(matrix(nrow=length(unique(data$ID)),ncol=0))
  test_table <- freq_table(feature=feature, data=data, by=by, token=token, i=1, pur="test")
  test_id <- test_table[,1]
  
  for(i in c(1:9)){
    test_table <- freq_table(feature=feature, data=data, by=by, token=token, i=i, pur="test")
    test_data <- test_table[,-1]
    #test_data <- ifelse(test_data==0, 0, 1) #uncomment this if you want binary naive-basyean classification
    
    temp <- as.matrix(test_data) %*% as.matrix(nb_model[,i])
    result <- data.frame(cbind(result, temp))
  }
  
  result <- t(apply(result, 1, softmax))
  result <- data.frame(cbind(as.numeric(test_id), result))
  names(result) <- c("ID", as.character(c(1:9)))
  
  return(result)
}


#완료된 freq_table은 svm에도 바로 사용할 수 있다. 시간이 조금 걸려도 svm까진 금방이니까 확해보자.

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

##multi-class xgboost
multi_xgboost <- function(feature=feature, tr_data=tr_data, te_data=te_data, by=by, token=token, params=param){
  #feature=class_word; tr_data=tr_word_data; te_data=te_word_data; by="n"; token="word"; params=param
  
  tr_dcg <- data.frame(sort(as.numeric(unique(tr_data$ID))))
  names(tr_dcg) <- "ID"
  te_dcg <- data.frame(sort(as.numeric(unique(te_data$ID))))
  names(te_dcg) <- "ID"
  
  for(i in c(1:9)){
    trn <- freq_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="test")
    tes <- freq_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
    
    trn <- trn %>% select(names(trn)[!names(trn) %in% intersect(names(tr_dcg), names(trn))])
    tes <- tes %>% select(names(tes)[!names(tes) %in% intersect(names(te_dcg), names(tes))])
    
    tr_dcg <- data.frame(cbind(tr_dcg, trn))
    te_dcg <- data.frame(cbind(te_dcg, tes))
  }
  
  tr_len <- trxt %>%
    mutate(text_len=log10(str_length(text))) %>%
    select(ID, text_len)
  
  te_len <- text %>%
    mutate(text_len=log10(str_length(text))) %>%
    select(ID, text_len)
  
  tr_gene <- onehot_gene(trv, 30)
  te_gene <- onehot_gene(tev, 30)
  
  tr_dcg <- tr_dcg %>% 
    merge(data.frame(cbind(tr_len, tr_gene)), by="ID")
  
  te_dcg <- te_dcg %>% 
    merge(data.frame(cbind(tr_len, tr_gene)), by="ID")
  
  
  tr_dcg <- tr_dcg %>%
    merge(trv, by="ID") %>%
    select(-Gene, -Variation) %>%
    mutate(ID=as.numeric(ID)) %>%
    arrange(ID)
  
  trn_lab <- tr_dcg$Class-1
  trn_data <- as(as.matrix(tr_dcg %>% select(-ID, -Class)), "dgCMatrix")
  
  trn_matrix <- xgb.DMatrix(data=trn_data, label=trn_lab)
  tes_data <- as(as.matrix(te_dcg %>% select(-ID)), "dgCMatrix")
  
  #cv.res <- xgb.cv(params=param, data=trn_matrix, nfold=5, early_stopping_rounds=3, nrounds=30)
  model <- xgboost(data=trn_matrix, nrounds=100, params=param, verbose=1)
  xgb_temp <- predict(model, tes_data)
  xgb_result <- matrix(xgb_temp, nrow = 9, ncol=length(xgb_temp)/9)
  xgb_result <- data.frame(cbind(te_dcg$ID, t(xgb_result)))
  names(xgb_result) <- c("ID", c(1:9))
  
  return(xgb_result)
}

#checking the result with validation set
max_class <- function(x){
  temp_id <- x[,1]
  temp <- apply(x[,-1], 1, function(y){ return(names(y)[which(y==max(y))][1]) })
  temp <- data.frame(cbind(temp_id, unlist(temp)))
  names(temp) <- c("ID","Class")
  
  return(temp)
}
result_table <- function(pred_result, te_label){
  res <- max_class(pred_result)
  res$Class <- factor(res$Class, levels = c(1:9))
  print(table(res$Class, te_label$Class, dnn=c("predicted","actual")))
  print(table(res$Class==te_label$Class))
  
  result_table <- data.frame(as.matrix(table(res$Class, te_label$Class, dnn=c("predicted","actual")), ncol=9))
  res$Class
  precision_recall <- result_table %>%
    group_by(predicted) %>%
    mutate(pre_sum=sum(Freq)) %>%
    ungroup() %>%
    group_by(actual) %>%
    mutate(act_sum=sum(Freq)) %>%
    ungroup() %>%
    filter(predicted==actual) %>%
    mutate(precision=Freq/pre_sum) %>%
    mutate(recall=Freq/act_sum) %>%
    mutate(Class=actual) %>%
    select(Class, precision, recall)
  
  #print(precision_recall)
  
  return(precision_recall)
}

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
#----------------#naive_basyean model#----------------#
#model using word & n
nb_model <- naive_baysean(feature=class_word, data=tr_word_data, by="n", token="word")
nb_result <- nb_predict(nb_model, feature=class_word, data=te_word_data, by="n", token="word")
temp <- result_table(nb_result, te_word_label) #accuracy : 0.3925
result_compare <- cbind(result_compare, temp[,-1])

#model using word & tf_idf
nb_model <- naive_baysean(feature=class_word_tf, data=tr_word_data, by="tf_idf", token="word")
nb_result <- nb_predict(nb_model, feature=class_word_tf, data=te_word_data, by="tf_idf", token="word")
temp <- result_table(nb_result, te_word_label) #accuracy : 0.1373
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & n
nb_model <- naive_baysean(feature=class_bigram, data=tr_bigram_data, by="n", token="bigram")
nb_result <- nb_predict(nb_model, feature=class_bigram, data=te_bigram_data, by="n", token="bigram")
temp <- result_table(nb_result, te_bigram_label) #accuracy : 0.3951
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & tf_idf
nb_model <- naive_baysean(feature=class_bigram_tf, data=tr_bigram_data, by="tf_idf", token="bigram")
nb_result <- nb_predict(nb_model, feature=class_bigram_tf, data=te_bigram_data, by="tf_idf", token="bigram")
temp <- result_table(nb_result, te_bigram_label) #accuracy : 0.3925
result_compare <- cbind(result_compare, temp[,-1])


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


#----------------#xgboost model#----------------#
param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              max_depth = 20,
              eta = 0.05,
              gamma = 0.01, 
              subsample = 0.9)

#model using word & n
xgb_result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", params=param)
temp <- result_table(xgb_result, te_word_label) #accuracy : 0.6244
result_compare <- cbind(result_compare, temp[,-1])

#model using word & tf_idf
xgb_result <- multi_xgboost(feature=class_word_tf, tr_data=tr_word_data, te_data=te_word_data, by="tf_idf", token="word", params=param)
temp <- result_table(xgb_result, te_word_label) #accuracy : 0.5432
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & n
xgb_result <- multi_xgboost(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", params=param)
temp <- result_table(xgb_result, te_bigram_label) #accuracy : 0.6146
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & tf_idf
xgb_result <- multi_xgboost(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", params=param)
temp <- result_table(xgb_result, te_bigram_label) #accuracy : 0.5664
result_compare <- cbind(result_compare, temp[,-1])

#ncol(result_compare)
names(result_compare) <- c("Class","nb_wd_n_precision","nb_wd_n_recall","nb_wd_tf_precision","nb_wd_tf_recall","nb_bg_n_precision","nb_bg_n_recall","nb_bg_tf_precision","nb_bg_tf_recall",
                           "svm_wd_n_precision","svm_wd_n_recall","svm_wd_tf_precision","svm_wd_tf_recall","svm_bg_n_precision","svm_bg_n_recall","svm_bg_tf_precision","svm_bg_tf_recall",
                           "xgb_wd_n_precision","xgb_wd_n_recall","xgb_wd_tf_precision","xgb_wd_tf_recall","xgb_bg_n_precision","xgb_bg_n_recall","xgb_bg_tf_precision","xgb_bg_tf_recall")

result_compare <- as.data.frame(t(result_compare[,-1]))
names(result_compare) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
View(result_compare)
write.csv(result_compare, file="result_compare.csv",row.names=FALSE)
#word와 bigram을 모두 feature로 가지는 xgb 모델을 만들었지만, 유의미한 변화는 없었다.
#다른 변수가 필요하다.



####ensemble model start
{
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
}



#####ensemble model(1) - RIPPER application
#model using bigram & tf_idf
ripper_result <- data.frame(matrix(ncol=10, nrow=0))
id_list <- vector()

#using xgb.bg.tf to classify Class 9
i = 9
result <- multi_xgboost(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
names(ripper_result) <- names(result)
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.bg.n to classify Class 7
i = 7
result <- multi_xgboost(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.wd.n to classify Class 6
i = 6
result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.bg.n to classify Class 4
i = 4
result <- multi_xgboost(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.bg.tf to classify Class 5
i = 5
result <- multi_xgboost(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.wd.n to classify Class 1
i = 1
result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.wd.n to classify Class 2
i = 2
result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using svm.wd.n to classify Class 3
i = 3
result <- multi_class_svm(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", kern="rbfdot")
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#using xgb.bg.n to classify Class 8
result <- multi_xgboost(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

#classify residual id
result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", params=param)
names(result) <- names(ripper_result)
ripper_result <- data.frame(rbind(ripper_result, result[which(!(result[,1] %in% id_list)),]))
id_list <- c(id_list, result[which(!(result[,1] %in% id_list) & (max.col(result[,-1])==i)),]$ID)

names(ripper_result) <- c("ID", c(1:9))
ripper_result <- ripper_result %>%
  arrange(ID)
result_table(ripper_result, te_word_label)

result_table(temp, qq)
#temp <- ripper_result











select(names(t1)[!names(t1) %in% intersect(names(tempt), names(t1))])
?with
??model.matrix

with(trv, head)

length(unique(qq$Gene))


       
View(t(result_compare[,seq(2, 25, 2)]))
View(t(result_compare[,seq(3, 25, 2)]))



#names(xgb_result) <- c("ID","class1","class2","class3","class4","class5","class6","class7","class8","class9")
#write.csv(xgb_result, file="xgb_result.csv",row.names=FALSE)


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


















