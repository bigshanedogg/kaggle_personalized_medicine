require(stats)
require(dplyr)
require(stats)
require(dplyr)
require(stringr)
require(tidyr)
require(tidytext)
require(tibble)
require(SnowballC)

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
tr_20_word <- top_word(trxt, 20) 

##word token 생성
tr_word_token <- trxt %>% 
  unnest_tokens(word, text) %>%
  mutate(word=wordStem(word)) %>%
  count(ID, word) %>% 
  merge(trv, by="ID") %>%
  select(ID, word, n, Class)

tr_word_token <-  tr_word_token %>% 
  filter(!word %in% tr_20_word$word) %>%
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
  filter(!word %in% tr_20_word$word) %>%
  filter(!word %in% stop_words$word) %>% 
  filter(!word %in% word_filter$word) %>%
  merge(tev, by="ID") %>%
  select(ID, word, n)

head(class_word)
head(tr_word_token)
#head(te_word_token)

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

head(class_bigram)
head(tr_bigram_token)
head(te_bigram_token)
#head(class_word)
head(tr_word_token)
#head(te_word_token)





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
multi_class_svm <- function(feature=feature, tr_data=tr_data, te_data=te_data, by=by, token=token, kern=kern){

  result <- data.frame(matrix(nrow=length(unique(te_data$ID)),ncol=0))
  test_id <- sort(as.numeric(unique(te_data$ID)))
  for(i in c(1:9)){
    svm_train <- svm_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="train")
    svm_test <- svm_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
    svm_test <- svm_test[,(names(svm_test) %in% names(svm_train))]
    
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
multi_xgboost <- function(feature=feature, tr_data=tr_data, te_data=te_data, by=by, token=token, param=param){

  result <- data.frame(matrix(nrow=length(unique(te_data$ID)), ncol=0))
  
  for(i in c(1:9)){
    trn <- freq_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="test")
    tes <- freq_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
    test_id <- tes[,1]
    
    trn <- trn %>%
      merge(trv, by="ID") %>%
      select(-Gene, -Variation) %>%
      mutate(ID=as.numeric(ID)) %>%
      arrange(ID)
    
    trn_lab <- trn[,22]
    trn_lab[which(trn_lab!=i)] <- 0
    trn_lab[which(trn_lab==i)] <- 1
    
    trn_data <- as(as.matrix(trn[,c(-1,-22)]), "dgCMatrix")
    
    #cv.res <- xgb.cv(data=trn_data, label=trn_lab, nfold=5, early_stopping_rounds=3, nrounds=30, objective="binary:logistic")
    #model <- xgboost(data=trn_data, label=trn_lab, nrounds=as.numeric(cv.res[7]), objective="binary:logistic")
    #cv.res <- xgb.cv(data=trn_data, label=trn_lab, nfold=5, early_stopping_rounds=3, nrounds=30, params=param)
    model <- xgboost(data=trn_data, label=trn_lab, nrounds=30, params=param)
    
    tes_data <- as(as.matrix(tes[,-1]), "dgCMatrix")
    xgb_temp <- predict(model, tes_data)
    xgb_temp <- data.frame(matrix(xgb_temp, ncol=2, byrow=FALSE))
    #result <- data.frame(cbind(result, xgb_temp))
    result <- data.frame(cbind(result, xgb_temp[,2]))
  }
  
  result <- t(apply(result, 1, softmax))
  result <- data.frame(cbind(test_id, result))
  names(result) <- c("ID", as.character(c(1:9)))
  
  return(result)
}
#xgb_result <- multi_xgboost(feature=class_word, tr_data=tr_word_token, te_data=te_word_token, by="n", token="word")

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
temp <- result_table(svm_result, te_word_label) #accuracy : 0.5646
result_compare <- cbind(result_compare, temp[,-1])

#model using word & tf_idf
svm_result <- multi_class_svm(feature=class_word_tf, tr_data=tr_word_data, te_data=te_word_data, by="tf_idf", token="word", kern="rbfdot")
temp <- result_table(svm_result, te_word_label) #accuracy : 0.3006
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & n
svm_result <- multi_class_svm(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", kern="rbfdot")
temp <- result_table(svm_result, te_bigram_label) #accuracy : 0.5414
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & tf_idf
svm_result <- multi_class_svm(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", kern="rbfdot")
temp <- result_table(svm_result, te_bigram_label) #accuracy : 0.4549
result_compare <- cbind(result_compare, temp[,-1])


#----------------#xgboost model#----------------#
param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 2,
              max_depth = 8,
              eta = 0.05,
              gamma = 0.01, 
              subsample = 0.9)

#model using word & n
xgb_result <- multi_xgboost(feature=class_word, tr_data=tr_word_data, te_data=te_word_data, by="n", token="word", param=param)
temp <- result_table(xgb_result, te_word_label) #accuracy : 0.0990
result_compare <- cbind(result_compare, temp[,-1])

#model using word & tf_idf
xgb_result <- multi_xgboost(feature=class_word_tf, tr_data=tr_word_data, te_data=te_word_data, by="tf_idf", token="word", param=param)
temp <- result_table(xgb_result, te_word_label) #accuracy : 0.1445
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & n
xgb_result <- multi_xgboost(feature=class_bigram, tr_data=tr_bigram_data, te_data=te_bigram_data, by="n", token="bigram", param=param)
temp <- result_table(xgb_result, te_bigram_label) #accuracy : 0.1409
result_compare <- cbind(result_compare, temp[,-1])

#model using bigram & tf_idf
xgb_result <- multi_xgboost(feature=class_bigram_tf, tr_data=tr_bigram_data, te_data=te_bigram_data, by="tf_idf", token="bigram", param=param)
temp <- result_table(xgb_result, te_bigram_label) #accuracy : 0.1409
result_compare <- cbind(result_compare, temp[,-1])

#ncol(result_compare)
names(result_compare) <- c("Class","nb_wd_n_precision","nb_wd_n_recall","nb_wd_tf_precision","nb_wd_tf_recall","nb_bg_n_precision","nb_bg_n_recall","nb_bg_tf_precision","nb_bg_tf_recall",
                           "svm_wd_n_precision","svm_wd_n_recall","svm_wd_tf_precision","svm_wd_tf_recall","svm_bg_n_precision","svm_bg_n_recall","svm_bg_tf_precision","svm_bg_tf_recall",
                           "xgb_wd_n_precision","xgb_wd_n_recall","xgb_wd_tf_precision","xgb_wd_tf_recall","xgb_bg_n_precision","xgb_bg_n_recall","xgb_bg_tf_precision","xgb_bg_tf_recall")

View(t(result_compare[,seq(2, 25, 2)]))
View(t(result_compare[,seq(3, 25, 2)]))
#xgb.plot.tree(feature_names = names(aa)[-1], model = model)

feature=class_word
tr_data=tr_word_data
te_data=te_word_data
by="n"
token="word"

tr_dcg <- data.frame(sort(as.numeric(unique(tr_data$ID))))
names(tr_dcg) <- "ID"
te_dcg <- data.frame(sort(as.numeric(unique(te_data$ID))))
names(te_dcg) <- "ID"

for(i in c(1:9)){
  trn <- freq_table(feature=feature, data=tr_data, by=by, token=token, i=i, pur="test")
  tes <- freq_table(feature=feature, data=te_data, by=by, token=token, i=i, pur="test")
  tr_dcg <- data.frame(cbind(tr_dcg, trn[,-1]))
  te_dcg <- data.frame(cbind(te_dcg, tes[,-1]))
}

tr_dcg <- tr_dcg %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  mutate(ID=as.numeric(ID)) %>%
  arrange(ID)

trn_lab <- tr_dcg$Class-1
trn_data <- as(as.matrix(tr_dcg[,c(-1,-182)]), "dgCMatrix")
#왜 num_class가 9이면 안되는지, num_class의 의미가 뭔지 제대로 확인하고 multi model로 다시 해보자.
#상식적으로 xgb의 지금 퍼포먼스는 납득이 안간다.
param <- list(objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              max_depth = 20,
              eta = 0.05,
              gamma = 0.01, 
              subsample = 0.9)

trn_matrix <- xgb.DMatrix(data=trn_data, label=trn_lab)
tes_data <- as(as.matrix(te_dcg[,-1]), "dgCMatrix")
#cv.res <- xgb.cv(params=param, data=trn_matrix, nfold=5, early_stopping_rounds=3, nrounds=30)
model <- xgboost(data=trn_matrix, nrounds=100, params=param, verbose=1)
xgb_temp <- predict(model, tes_data)
xgb_result <- matrix(xgb_temp, nrow = 9, ncol=length(xgb_temp)/9)
xgb_result <- data.frame(cbind(te_dcg$ID, t(xgb_result)))
names(xgb_result) <- c("ID", c(1:9))

result_table(xgb_result, te_word_label)

head(trn_data)

#names(result) <- c("ID", c("class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9"))
#write.csv(result, file="xgb_result.csv",row.names=FALSE)

head(te_dcg)
aa <- tr_dcg[,-1]
aa$Class <- as.factor(aa$Class)
svm_model <- ksvm(Class~., data=aa, kernel="rbfdot", prob.model=TRUE)
svm_temp <- predict(svm_model, te_dcg[,-1], type="probabilities")
head(max_class(svm_temp))
?max.col

table(max.col(svm_temp), te_word_label$Class)
table(max.col(svm_temp)==te_word_label$Class)
head(te_word_label)

result_table(svm_result, te_word_label)

















