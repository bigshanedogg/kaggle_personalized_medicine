#for basic data manipuldation
require(stats)
require(plyr)
require(dplyr) 
require(lubridate) #for processing time-series data
require(geosphere)
require(reshape)
require(tibble)
require(stringr)
require(SnowballC)
require(tidytext)
require(tidyr)

#for basic visualization
require(extrafont) #for using 'Helvetica'
require(RColorBrewer)
require(ggplot2) #basic visualization
require(GGally)
require(grid)

#for mapdata
require(maps)
require(mapdata)
require(leaflet) #real-time mapping

#for k-means, k-nn, and xgboost model
require(cluster)
require(class)
require(xgboost)

#multiplot function
multiplot <- function(..., plotlist = NULL, file, cols = 1, layout = NULL) {
  require(grid)
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  if (is.null(layout)) {
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))}
  if (numPlots == 1) { print(plots[[1]])
  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col)) }}}
ezLev <- function(x,new_order){
  for(i in rev(new_order)){
    x=relevel(x,ref=i)
  }
  return(x)
}
ggcorplot <- function(data,var_text_size,cor_text_limits){
  # normalize data
  for(i in 1:length(data)){
    data[,i]=(data[,i]-mean(data[,i]))/sd(data[,i])
  }
  # obtain new data frame
  z=data.frame()
  i = 1
  j = i
  while(i<=length(data)){
    if(j>length(data)){
      i=i+1
      j=i
    }else{
      x = data[,i]
      y = data[,j]
      temp=as.data.frame(cbind(x,y))
      temp=cbind(temp,names(data)[i],names(data)[j])
      z=rbind(z,temp)
      j=j+1
    }
  }
  names(z)=c('x','y','x_lab','y_lab')
  z$x_lab = ezLev(factor(z$x_lab),names(data))
  z$y_lab = ezLev(factor(z$y_lab),names(data))
  z=z[z$x_lab!=z$y_lab,]
  #obtain correlation values
  z_cor = data.frame()
  i = 1
  j = i
  while(i<=length(data)){
    if(j>length(data)){
      i=i+1
      j=i
    }else{
      x = data[,i]
      y = data[,j]
      x_mid = min(x)+diff(range(x))/2
      y_mid = min(y)+diff(range(y))/2
      this_cor = cor(x,y)
      this_cor.test = cor.test(x,y)
      this_col = ifelse(this_cor.test$p.value<.05,'<.05','>.05')
      this_size = (this_cor)^2
      cor_text = ifelse(
        this_cor>0
        ,substr(format(c(this_cor,.123456789),digits=2)[1],2,4)
        ,paste('-',substr(format(c(this_cor,.123456789),digits=2)[1],3,5),sep='')
      )
      b=as.data.frame(cor_text)
      b=cbind(b,x_mid,y_mid,this_col,this_size,names(data)[j],names(data)[i])
      z_cor=rbind(z_cor,b)
      j=j+1
    }
  }
  names(z_cor)=c('cor','x_mid','y_mid','p','rsq','x_lab','y_lab')
  z_cor$x_lab = ezLev(factor(z_cor$x_lab),names(data))
  z_cor$y_lab = ezLev(factor(z_cor$y_lab),names(data))
  diag = z_cor[z_cor$x_lab==z_cor$y_lab,]
  z_cor=z_cor[z_cor$x_lab!=z_cor$y_lab,]
  #start creating layers
  points_layer = layer(
    geom = 'point'
    , data = z
    , mapping = aes(
      x = x
      , y = y
    )
  )
  lm_line_layer = layer(
    geom = 'line'
    , geom_params = list(colour = 'red')
    , stat = 'smooth'
    , stat_params = list(method = 'lm')
    , data = z
    , mapping = aes(
      x = x
      , y = y
    )
  )
  lm_ribbon_layer = layer(
    geom = 'ribbon'
    , geom_params = list(fill = 'green', alpha = .5)
    , stat = 'smooth'
    , stat_params = list(method = 'lm')
    , data = z
    , mapping = aes(
      x = x
      , y = y
    )
  )
  cor_text = layer(
    geom = 'text'
    , data = z_cor
    , mapping = aes(
      x=y_mid
      , y=x_mid
      , label=cor
      , size = rsq
      , colour = p
    )
  )
  var_text = layer(
    geom = 'text'
    , geom_params = list(size=var_text_size)
    , data = diag
    , mapping = aes(
      x=y_mid
      , y=x_mid
      , label=x_lab
    )
  )
  f = facet_grid(y_lab~x_lab,scales='free')
  o = opts(
    panel.grid.minor = theme_blank()
    ,panel.grid.major = theme_blank()
    ,axis.ticks = theme_blank()
    ,axis.text.y = theme_blank()
    ,axis.text.x = theme_blank()
    ,axis.title.y = theme_blank()
    ,axis.title.x = theme_blank()
    ,legend.position='none'
  )
  
  size_scale = scale_size(limits = c(0,1),to=cor_text_limits)
  return(
    ggplot()+
      points_layer+
      lm_ribbon_layer+
      lm_line_layer+
      var_text+
      cor_text+
      f+
      o+
      size_scale
  )
}
#--------------------------------------------------------#
##Import data and breif check


#load data
trv <- data.frame(read.csv("../raw_data/training_variants"))
tev <- data.frame(read.csv("../raw_data/test_variants"))

temp <- readLines("training_text")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
trxt <- data_frame(ID=temp[,1], text=temp[,2])

temp <- readLines("test_text.csv")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
text <- data_frame(ID=temp[,1], text=temp[,2])

#brief check
glimpse(trv)
glimpse(trxt)

glimpse(tev)
glimpse(text)

#null check
sum(is.na(trv))
sum(is.na(tev))
#--------------------------------------------------------#

#--------------------------------------------------------#
##Check data
#Let's check some variables - Gene&Variation by length  before visulization
##- If unique length of those variables factors are not low, It may have not much influence the model learning because nrow() of training and test are not high (3321, 986)
length(unique(trv$Gene))
length(unique(tev$Gene))
length(intersect(unique(tev$Gene), unique(trv$Gene)))
length(union(unique(tev$Gene), unique(trv$Gene)))

length(unique(trv$Variation))
length(unique(tev$Variation))
length(intersect(unique(tev$Variation), unique(trv$Variation)))
length(union(unique(tev$Variation), unique(trv$Variation)))
#Variation seems not meaningful variable for learning yet.

trv %>%
  group_by(Gene) %>%
  count() %>%
  summary()

trv %>%
  group_by(Variation) %>%
  count() %>%
  summary()

trv %>%
  group_by(Class) %>%
  count() %>%
  summary()

gene_freq <- trv %>% #check Gene frequency
  group_by(Gene) %>%
  count() %>%
  arrange(desc(n)) %>%
  head(n=20) %>%
  ggplot(aes(reorder(Gene, n),n , fill=Gene)) + 
  geom_col() + 
  geom_text(aes(label=n), size = 3, position = position_stack(vjust = 0.5)) +
  coord_flip() +
  theme_gray(base_family = "Helvetica") +
  theme(legend.position="none") + 
  labs(title="histogram")

var_freq <- trv %>% #check Variation frequency 
  group_by(Variation) %>%
  count() %>%
  arrange(desc(n)) %>%
  head(n=20) %>%
  ggplot(aes(reorder(Variation, n),n , fill=Variation)) + 
  geom_col() + 
  geom_text(aes(label=n), size = 3, position = position_stack(vjust = 0.5)) +
  coord_flip() +
  theme_gray(base_family = "Helvetica") +
  theme(legend.position="none") + 
  labs(title="histogram")

class_freq <- trv %>%
  group_by(Class) %>%
  count() %>%
  ggplot(aes(reorder(Class, -as.numeric(Class)),n , fill=Class)) + 
  geom_col() + 
  geom_text(aes(label=n), size = 3, color="white", position = position_stack(vjust = 0.5)) +
  coord_flip() +
  theme_gray(base_family = "Helvetica") +
  theme(legend.position="none") + 
  labs(title="histogram")

layout <- matrix(c(1,2,3),1,3,byrow=TRUE)
multiplot(gene_freq, var_freq, class_freq, layout=layout)

#In Gene feature, It shows not bad distribution compared with Variation. we can try to use it.
#In Variation feature, Almost classes are observed once or twice except few classes.
#In label(Class), we can check 3/8/9 class in detail while checking word & bigram  because those shows low frequency.
#The distribution of other classes seems not bad. (concetrated not too much to specific class)

#Gene/Variation frequency comparison of training & test set
#used head(n=10) instead of top_n(10, n) because top_n() include all the same frequency rows
tr_10_gene <- trv %>%
  count(Gene) %>%
  arrange(desc(n)) %>%
  head(n=10) %>%
  mutate(div="tr")

te_10_gene <- tev %>%
  count(Gene) %>%
  arrange(desc(n)) %>%
  head(n=10) %>%
  mutate(div="te")

gene_compare <- data.frame(rbind(te_10_gene, tr_10_gene)) %>%
  ggplot(aes(x=Gene, y=n, group=div, fill=div, color=div)) + 
  geom_line()

tr_10_var <- trv %>%
  count(Variation) %>%
  arrange(desc(n)) %>%
  head(n=10) %>%
  mutate(div="tr") 

te_10_var <- tev %>%
  count(Variation) %>%
  arrange(desc(n)) %>%
  head(n=10) %>%
  mutate(div="te")

var_compare <- data.frame(rbind(te_10_var, tr_10_var)) %>%
  ggplot(aes(x=Variation, y=n, group=div, fill=div, color=div)) + 
  geom_line()

layout <- matrix(c(1,2),1,2,byrow=TRUE)
multiplot(gene_compare, var_compare)
#대부분의 Variation, Gene은 빈도수가 3 이하이므로 group_by(Class) top_n(10, n)은 의미가 없다. 
#따라서 전체 빈도수 top 10의 Class별 분포를 확인한다.
#(빈도수 차이는 nrow의 차이로 인한 것이고) 앞선 length 비교에서 확인했듯이 training과 test의 Variation의 교집합이 적고 Gene의 교집합이 비교적 클 것으로 예상했었는데,
#실제로 Variation은 Gene에 비해 교집합이 적다. 그리고 length 비교에서 tr, te set의 unique합이 nrow합과 유사한 것으로 보아 학습에 큰 의미가 있을 것 같지 않다. 
#반면, Gene의 top 10에 꽤 교집합이 있는 것으로 보이므로 실제 factor set 구성의 차이가 크진 않고, tr, te set의 unique합이 nrow합과 차이가 있으므로 학습에 유의미할 것이다.


gene_tr_class <- trv %>%
  filter(Gene %in% as.character(tr_10_gene$Gene)) %>%
  ggplot(aes(Gene)) +
  geom_bar() +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  facet_wrap(~ Class)

var_tr_class <- trv %>%
  filter(Variation %in% as.character(tr_10_var$Variation)) %>%
  ggplot(aes(Variation)) +
  geom_bar() +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  facet_wrap(~Class)

layout <- matrix(c(1,2),1,2,byrow=TRUE)
multiplot(gene_tr_class, var_tr_class)
#Class별 Variation과 Gene의 분포가 꽤 다른 것으로 보인다. sparse matrix가 될 것으로 예상되지만, 두 변수를 더 살펴볼 필요가 있다.
#Class 3,5,9 are removed because they don't have a row include top 10 Variation class 



#--------------------------------------------------------#

#--------------------------------------------------------#
##Word & Bigram visualization
trxt %>%
  mutate(text_len=str_length(text)) %>%
  summary()

trxt %>%
  mutate(text_len=str_length(text)) %>%
  filter(text_len<=100) %>%
  select(ID, text, text_len)

text %>%
  mutate(text_len=str_length(text)) %>%
  summary()

text %>%
  mutate(text_len=str_length(text)) %>%
  filter(text_len<=100) %>%
  select(ID, text, text_len)
#we can check there are 5 rows without description text in training set, while test set are not.

trxt %>%
  merge(trv, by="ID") %>%
  select(ID, text, Class) %>%
  mutate(text_len=str_length(text)) %>%
  ggplot(aes(text_len, fill=as.factor(Class))) +
  geom_histogram(bins=50) + 
  facet_wrap(~Class)
#we can check that Class 1,4,7 has more quantitative information compared with other, especially Class 3,8,9.
#we already checked Class 3,8,9 has less frequency remarkablely than other.
#text_len can be meaningful variables to classify Class 3,8,9 and others.

tr_word_token <- trxt %>% merge(trv, by="ID") %>%
  select(ID, text, Class) %>%
  unnest_tokens(word, text) %>%
  mutate(word=wordStem(word))

te_word_token <- text %>% 
  unnest_tokens(word, text) %>%
  mutate(word=wordStem(word))

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
test_top_20 <- top_word(text, 20)  
intersect(top_20_word$word, test_top_20$word)
#Both top_10_word of training and test set are same and those top 10 word looks like stop_words such as "and", "the", "of" etc.

tr_word_token %>%
  filter(word %in% top_20_word$word) %>%
  count(Class, word) %>%
  ggplot(aes(x=word, y=n, fill=as.factor(Class))) +
  geom_bar(stat="identity") +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  facet_wrap(~ Class)
#top_20_word distribution by Class looks similar, so we'd better remove those words.

data("stop_words")
tr_word_token <-  tr_word_token %>% 
  filter(!word %in% top_20_word$word) %>%
  filter(!word %in% stop_words$word) 
#Let's remove top_20_word and stop_words at once.

word_filter <- tr_word_token %>%
  count(ID, word) %>%
  bind_tf_idf(word, ID, n) %>%
  select(word, tf_idf) %>%
  unique() %>%
  arrange(tf_idf) %>% 
  select(word) %>%
  unique() %>%
  head(n=30)

word_filter$word

tr_word_token %>%
  filter(word %in% word_filter$word) %>%
  count(Class, word) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  ggplot(aes(x=word, y=n, fill=as.factor(Class))) +
  geom_bar(stat="identity") +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  facet_wrap(~ Class) +
  coord_flip()

class_word <- tr_word_token %>%
  filter(!word %in% word_filter$word) %>%
  count(Class, word) %>%
  arrange(Class, desc(n)) %>%
  group_by(Class) %>%
  top_n(20, n) 

class_word %>%
  group_by(Class) %>% 
  top_n(20, n) %>%
  arrange(word) %>%
  ggplot(aes(word, n, fill = as.factor(Class))) +
  geom_col() +
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()


#Words showing low tf_idf can be unsignificant words filtered by tf-idf because low tf_idf means that they are not discriminant factor
#We can check distribution of filtered words are similar in all Class in graph
#(Class 3,8,9 show less frequency of those words than other because they have less frequecny and text_length as we know)
#Made these steps by function because we will use filtered word as well as stop_words we used once in refining word of test set, and bigram visualization


#Word set by Class seems quite different except some words such as "tumor", "variant" etc
#Those common words have different frequencies by Class, which mean that they can be discriminative variables in modeling.
#Below code shows different frequency of "tumor" by Class.
tr_word_token %>%
  count(ID, word) %>%
  filter(word=="tumor") %>%
  merge(trv, by="ID") %>%
  select(Class, word, n) %>%
  group_by(Class) %>%
  mutate(t_m = mean(n)) %>%
  select(Class, word, t_m) %>%
  unique()
head(tr_word_token)
#We extracted top 20 words per each Class by their frequencies.
#below code is for extracted top 20 words per each Class by tf_idf.
class_word_tf <- tr_word_token %>%
  filter(!word %in% word_filter$word) %>%
  count(ID, word) %>%
  bind_tf_idf(word, ID, n) %>%
  merge(trv, by="ID") %>%
  select(word, tf_idf, Class) %>%
  distinct(Class, word, .keep_all=TRUE) %>%
  group_by(Class) %>%
  top_n(20, tf_idf) %>% 
  arrange(Class, desc(tf_idf))

tr_word_token %>%
  filter(word %in% class_word_tf$word) %>%
  count(ID, word) %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(word, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()
#We can see that the factor set - extracted words and their frequencies are quite different with the former.
#Let's keep both and check in validation stage, because there's not enough grounds to choose now.

#Class별 word_n, sentence_n, text_len을 확인한다.
#우선 변수 정제
word_n <- trxt %>%
  unnest_tokens(word, text, token="words") %>%
  count(ID) %>%
  mutate(word_n = n) %>%
  select(ID, word_n)

sentence_n <- trxt %>%
  unnest_tokens(sentence, text, token="sentences") %>%
  count(ID) %>%
  mutate(sentence_n = n) %>%
  select(ID, sentence_n)

tr_feature <- trv %>%
  merge(trxt, by="ID") %>%
  mutate(text_len = str_length(text)) %>%
  merge(word_n, by="ID") %>%
  merge(sentence_n, by="ID") %>%
  select(ID, Gene, Variation, text_len, word_n, sentence_n, Class)

feature_refining <- function(x, y){ 
  #x : trxt, text
  #y : trv, tev
  
  word_n <- x %>%
    unnest_tokens(word, text, token="words") %>%
    count(ID) %>%
    mutate(word_n = n) %>%
    select(ID, word_n)
  
  sentence_n <- x %>%
    unnest_tokens(sentence, text, token="sentences") %>%
    count(ID) %>%
    mutate(sentence_n = n) %>%
    select(ID, sentence_n)
  
  feature <- y %>%
    merge(x, by="ID") %>%
    mutate(text_len = str_length(text)) %>%
    merge(word_n, by="ID") %>%
    merge(sentence_n, by="ID") %>%
    select(ID, Gene, Variation, text_len, word_n, sentence_n)
  
  return(feature)
}
te_feature <- feature_refining(text, tev)

##visualizing text_len, word_n, sentence_n
#세 변수가 높은 상관관계를 지닐 것은 직관적으로 당연히 알 수 있지만, 시각화를 통해 확실히 짚고 넘어가보자.
text_len_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, text_len, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  scale_y_log10() + 
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  labs(title="text_len")

word_n_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, word_n, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  labs(title="word_n")

sentence_n_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, sentence_n, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  labs(title="sentence_n")

n_pairs <- tr_feature %>%
  select(text_len, word_n, sentence_n) %>%
  ggpairs()

layout <- matrix(c(1,2,3,4,4,4),2,3,byrow=TRUE)
multiplot(text_len_boxplot, word_n_boxplot, sentence_n_boxplot, n_pairs, layout=layout)
#역시나 마지막 pairs plot에서 확인할 수 있듯이 아주아주아주 높은 상관관계를 지니고 있다. 
#text_len의 분포가 조금 달라보이지만 별 의미가 없는 차이다. (text_len의 차이는 사용되는 단어의 길이 차이에 의한 것인데, 긴 단어라고 높은 정보량을 지니지 않기 때문)
#그러므로, 실제 모델링시 이 중에선 word_n 변수만 사용한다.

head(stop_words$word)
head(top_20_word$word)

##bigram
#tr_word_token으로 bigram <- 생성
tr_bigram_token <- trxt %>% 
  select(ID, text) %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c('w1','w2'), sep=" ") %>%
  mutate(w1=wordStem(w1)) %>%
  mutate(w2=wordStem(w2)) %>%
  filter(!w1 %in% stop_words$word) %>%
  filter(!w2 %in% stop_words$word) %>%
  filter(!w1 %in% top_20_word$word) %>%
  filter(!w2 %in% top_20_word$word) %>%
  unite(bigram, w1, w2, sep=" ")

#below graph shows the bigram distribution by class without additional filtering
tr_bigram_token %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  count(Class, bigram) %>%
  group_by(Class) %>%
  top_n(10, n) %>%
  ungroup() %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free_y") +
  coord_flip()
#추가적인 필터링 없이 바이그램 필터만 봐도 충분히 달라진 게 보인다.
#(한눈에 확인하기 위해 fixed를 사용하면 세부 단어들은 보기 어렵지만 차이는 더 명확하게 보인다.) 
#팩터셋 구성도 다르고, 세부적인 분포도 다르다.
#무엇보다 여태까지 비슷한 패턴을 보였던 Class 8과 Class 9가 확연히 구분된다는 점. (오히려 Class 5와 Class 8이 유사한 패턴을 보인다.)
#이대로도 충분히 괜찮지만, wild type과 같이 전체에서 모두 관측되는 단어들을 제거해 정확도를 향상시키기 위해 추가로 필터링을 할 수 있을지 체크해본다.

#below code is for making bigram_filter as we made word_filter.
bigram_filter <- tr_bigram_token %>%
  count(ID, bigram) %>%
  bind_tf_idf(bigram, ID, n) %>%
  select(bigram, tf_idf) %>%
  unique() %>%
  arrange(tf_idf) %>% 
  select(bigram) %>%
  unique() %>%
  head(n=15)

bigram_filter$bigram

#below graph shows frequency of bigram in bigram_filter by Class to see bigram_filter is useful filter
#If distribution seems similar regardless of Class, The filter is useful.
tr_bigram_token %>%
  filter(bigram %in% bigram_filter$bigram) %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  count(Class, bigram) %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()
#fixed로 확인해보면 빈도수가 확연히 차이나는 것을 확인할 수 있다. 그러나, 이것은 기본적으로 빈도에 의한 것이다. 
#(빈도가 높은 클래스인 1,4,7의 경우 높지만 다른 클래스는 낮은 것을 통해 알 수 있다.)
#scales="free"로 확인해볼 경우, 분포 패턴이 유사함을 확인할 수 있다. 즉, bigram_filter는 유용한 필터일 가능성이 높아졌다. 

#below graph shows distribution of bigram filtered using bigram_filter
tr_bigram_token %>%
  filter(!bigram %in% bigram_filter$bigram) %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  count(Class, bigram) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()
#factor set이 상당히 상호배제적이다. 이 bigram set으로 모델링을 시도해볼 수 있겠다.
#다만 걱정되는 것은 일부 bigram의 관측빈도(n)이 낮다는 것이다. 
#예를 들어, Class 1의 빈도수가 568개인데 빈도수가 200 남짓한 "atrx loss"의 경우 훈련 데이터에서 모아봤을 땐 패턴을 이루어도, 실제 예측 단계에서 해당 bigram이 없을 확률이 높기 때문이다.
#이 경우는 factor set이 충분히 상호배제적이기 때문에 해당 bigram이 포함될 경우 확실히 분류될 확률이 높아지고, 그렇지 않을 경우 다른 bigram과 word로 정분류될 확률을 높이기 위해 모델링 전에 전체 feature 구성을 다시 한번 점검할 필요가 있다.

#below 'tbt_fted' is filtered using word_filter (seperate bigram, filter w1, w2 with word_filter we made once, and unite to bigram again)
tbt_fted <- tr_bigram_token %>%
  separate(bigram, c("w1","w2"), sep=" ") %>%
  filter(!w1 %in% word_filter$word) %>%
  filter(!w2 %in% word_filter$word) %>%
  unite(bigram, w1, w2, sep=" ")

tbt_fted %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  count(Class, bigram) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()
#사실 word_filter로 거른 factor set도 충분히 상호배제적이며, 괜찮은 패턴을 보인다.
#그러나, 이번엔 다음과 같은 2가지 이유에서 bigram_filter로 필터링하기로 한다.
#1) 일부 클래스의 n 빈도가 더 높은 facotr set으로 출력된다.(해당 클래스에서 그 bigram이 관측될 확률이 높아지는 것을 의미하기 때문이다.
#2) word_filter를 이용할 경우 "mutant", "cell" 등의 단어가 전부 사라진다. 즉, word feature와 bigram feature에서 "mutant"과 같은 단어를 모두 볼 수 없게 된다.
#그러나, word feature만 이용할 경우 "mutant"이라는 단어 자체는 모든 클래스에서 자주 관측되는 단어일 확률이 높지만, bigram의 형태에선 다른 단어 예를 들면 클래스 9의 "sf3b1 mutant"과 같이 다른 단어와 결합해서 더 많은 의미를 내포하는 경우가 있을 수 있기 때문이다.

tr_bigram_token <- tr_bigram_token %>%
  filter(!bigram %in% bigram_filter$bigram)

class_bigram <- tr_bigram_token %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation, -ID) %>%
  count(Class, bigram) %>%
  distinct(Class, bigram, .keep_all=TRUE) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  arrange(Class, desc(n))

#We extracted top 20 bigrams per each Class by their frequencies.
#below code is for extracted top 20 bigrams per each Class by tf_idf.
class_bigram_tf <- tr_bigram_token %>%
  merge(trv, by="ID") %>%
  select(ID, bigram, Class) %>%
  count(Class, bigram) %>%
  bind_tf_idf(bigram, Class, n) %>%
  select(bigram, tf_idf, Class) %>%
  distinct(Class, bigram, .keep_all=TRUE) %>%
  group_by(Class) %>%
  top_n(20, tf_idf) %>%
  arrange(Class, desc(tf_idf))


tr_bigram_token %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  group_by(Class) %>%
  filter(bigram %in% class_bigram_tf$bigram) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()
#We can see that the factor set - extracted bigrams and their frequencies are quite different with the former.
#Let's keep both and check in validation stage, because there's not enough grounds to choose now.

#n 또는 tf-idf가 중복되어 class당 feature 개수가 20을 넘어가는 행을 자름
trunc_feature <- function(x){
  temp <- head(x, n=0)
  names(temp) <- names(x)
  
  for(i in c(1:9)){
    c <- x %>%
      filter(Class==i)
    if(nrow(c)>20){
      c <- head(c, n=20)
    }
    
    temp <- rbind(temp, c)
  }
  return(temp)
}

class_word <- trunc_feature(class_word)
class_word_tf <- trunc_feature(class_word_tf)
class_bigram <- trunc_feature(class_bigram)
class_bigram_tf <- trunc_feature(class_bigram_tf)


#완료 #1) bigram 변수에 tf_idf로 bigram_filter 생성
#완료 #2) bigram 변수 생성시 word_filter로 filtering하여 생성

#완료 #생성된 bigram 변수 확인
#완료 #bigram_filter에 있는 단어를 facet_wrap으로 확인
#완료 #Class별로 10개 혹은 20개의 상위 bigram을 가지는 class_bigram 변수 생성
#완료 #class_bigram을 이용하여 bigram_filter에 없는 단어를 facet_wrap으로 확인

#완료 #class_word와 class_bigram의 tf_idf 버전을 만든뒤 앞선 2개의 class_word와 class_bigram과 함께 
#완료 #csv 파일로 만들어, naive_baysean으로 넘어간다.

#csv로 넘길 것들
#tr_feature #ID, Gene, Variation, text_len, word_n, senetence_n
#te_feature #ID, Gene, Variation, text_len, word_n, senetence_n
#class_word #top 20 words per each Class by frequency
#class_word_tf #top 20 words per each Class by tf_idf
#class_bigram #top 20 bigrams per each Class by frequency
#class_bigram_tf #top 20 bigrams per each Class by tf_idf
write.csv(tr_feature, file="../source/tr_feature.csv",row.names=FALSE)
write.csv(te_feature, file="../source/te_feature.csv",row.names=FALSE)
write.csv(class_word, file="../source/class_word.csv",row.names=FALSE)
write.csv(class_word_tf, file="../source/class_word_tf.csv",row.names=FALSE)
write.csv(class_bigram, file="../source/class_bigram.csv",row.names=FALSE)
write.csv(class_bigram_tf, file="../source/class_bigram_tf.csv",row.names=FALSE)
