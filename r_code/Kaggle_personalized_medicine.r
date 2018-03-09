
#for basic data manipuldation
require(stats)
require(plyr)
require(dplyr) 
require(lubridate) #for processing time-series data
require(geosphere)
require(reshape)
require(reshape2)
require(tibble)
require(stringr)
require(SnowballC)
require(tidytext)
require(tidyr)
require(onehot)

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

#load data
trv <- data.frame(read.csv("../raw_data/training_variants"))
tev <- data.frame(read.csv("../raw_data/test_variants.csv"))

temp <- readLines("../raw_data/training_text")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
trxt <- data_frame(ID=temp[,1], text=temp[,2])

temp <- readLines("../raw_data/test_text.csv")
temp <- str_split_fixed(temp[2:length(temp)], "\\|\\|",2)
text <- data_frame(ID=temp[,1], text=temp[,2])

cat("NULL rows\ntraining :", sum(is.na(trv)), "\ttest :", sum(is.na(tev)))

glimpse(trv)
glimpse(trxt)

glimpse(tev)
glimpse(text)

cat("*Number of Gene\ntraining_set :", length(unique(trv$Gene)), "\ttest_set :", length(unique(tev$Gene)),"\ttotal :", (length(unique(trv$Gene))+length(unique(tev$Gene))))
cat("\nIntersection :", length(intersect(unique(tev$Gene), unique(trv$Gene))), "\tUnion :", length(union(unique(tev$Gene), unique(trv$Gene))))
cat("\n\n*Number of Variation\ntraining_set :", length(unique(trv$Variation)), "\ttest_set :", length(unique(tev$Variation)),"\ttotal :", (length(unique(trv$Variation))+length(unique(tev$Variation))))
cat("\nIntersection :", length(intersect(unique(tev$Variation), unique(trv$Variation))), "\tUnion :", length(union(unique(tev$Variation), unique(trv$Variation))))

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

gene_tr_class <- trv %>%
  filter(Gene %in% as.character(tr_10_gene$Gene)) %>%
  ggplot(aes(Gene)) +
  geom_bar() +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class)

var_tr_class <- trv %>%
  filter(Variation %in% as.character(tr_10_var$Variation)) %>%
  ggplot(aes(Variation)) +
  geom_bar() +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~Class)

layout <- matrix(c(1,2),1,2,byrow=TRUE)
multiplot(gene_tr_class, var_tr_class)

gene_compare <- data.frame(rbind(te_10_gene, tr_10_gene)) %>%
  ggplot(aes(x=Gene, y=n, group=div, fill=div, color=div)) + 
  geom_line() + 
  theme_gray(base_family = "Helvetica")

var_compare <- data.frame(rbind(te_10_var, tr_10_var)) %>%
  ggplot(aes(x=Variation, y=n, group=div, fill=div, color=div)) + 
  geom_line() +
  theme_gray(base_family = "Helvetica")

layout <- matrix(c(1,2),1,2,byrow=TRUE)
multiplot(gene_compare, var_compare)

trxt %>%
  mutate(text_len=str_length(text)) %>%
  summary()

trxt %>%
  mutate(text_len=str_length(text)) %>%
  filter(text_len<=100) %>%
  select(ID, text, text_len)

trxt <- trxt %>%
  mutate(text_len=str_length(text)) %>%
  filter(text_len>100) %>%
  select(ID, text)

text %>%
  mutate(text_len=str_length(text)) %>%
  summary()

text %>%
  mutate(text_len=str_length(text)) %>%
  filter(text_len<=100) %>%
  select(ID, text, text_len)

trxt %>%
  merge(trv, by="ID") %>%
  select(ID, text, Class) %>%
  mutate(text_len=str_length(text)) %>%
  ggplot(aes(text_len, fill=as.factor(Class))) +
  geom_histogram(bins=50) + 
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~Class)

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

head(tr_feature)

text_len_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, text_len, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  scale_y_log10() + 
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  theme_gray(base_family = "Helvetica") +
  labs(title="text_len")

word_n_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, word_n, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  theme_gray(base_family = "Helvetica") +
  labs(title="word_n")

sentence_n_boxplot <- tr_feature %>%
  mutate(Class=as.factor(Class)) %>%
  ggplot(aes(Class, sentence_n, group=Class, fill=Class)) +
  geom_boxplot() +
  theme(legend.position="none") +
  coord_flip() + 
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  theme_gray(base_family = "Helvetica") +
  labs(title="sentence_n")

n_pairs <- tr_feature %>%
  select(text_len, word_n, sentence_n) %>%
  ggpairs() +
  theme_gray(base_family = "Helvetica")

layout <- matrix(c(1,2,3,4,4,4),2,3,byrow=TRUE)
multiplot(text_len_boxplot, word_n_boxplot, sentence_n_boxplot, n_pairs, layout=layout)

tr_word_token <- trxt %>% merge(trv, by="ID") %>%
  select(ID, text, Class) %>%
  unnest_tokens(word, text) %>%
  mutate(word=wordStem(word))

te_word_token <- text %>% 
  unnest_tokens(word, text) %>%
  mutate(word=wordStem(word))

top_word <- function(x, y){ #텍스트 데이터에서 y개의 top frequency 단어 추출
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
test_top_20 <- top_word(text, 20) %>% select(word) 

intersect(top_20_word$word, test_top_20$word)

tr_word_token %>%
  filter(word %in% top_20_word$word) %>%
  count(Class, word) %>%
  ggplot(aes(x=word, y=n, fill=as.factor(Class))) +
  geom_bar(stat="identity") +
  scale_y_log10() +
  theme(axis.text.x  = element_text(angle=90, vjust=0.5, size=7)) +
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class)

data("stop_words")

head(stop_words$word, n=20)

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
  head(n=20)

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
  theme_gray(base_family = "Helvetica") +
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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

tr_word_token %>%
  count(ID, word) %>%
  filter(word=="tumor") %>%
  merge(trv, by="ID") %>%
  select(Class, word, n) %>%
  group_by(Class) %>%
  mutate(t_m = mean(n)) %>%
  select(Class, word, t_m) %>%
  unique()

class_word_tf <- tr_word_token %>%
  filter(!word %in% word_filter$word) %>%
  count(ID, word) %>%
  bind_tf_idf(word, ID, n) %>%
  merge(trv, by="ID") %>%
  select(word, tf_idf, Class) %>%
  group_by(Class) %>%
  top_n(20, tf_idf) %>% 
  arrange(Class, desc(tf_idf))

tr_word_token %>%
  filter(word %in% class_word_tf$word) %>%
  count(ID, word) %>%
  merge(trv, by="ID") %>%
  select(-ID, -Gene, -Variation) %>%
  group_by(Class) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(word, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free_y") +
  coord_flip()

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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="fixed") +
  coord_flip()

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

tr_bigram_token %>%
  filter(bigram %in% bigram_filter$bigram) %>%
  merge(trv, by="ID") %>%
  select(-Gene, -Variation) %>%
  count(Class, bigram) %>%
  ggplot(aes(bigram, n, fill=as.factor(Class))) +
  geom_col() + 
  labs(x = NULL, y = "n") +
  theme(legend.position = "none") +
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

options(warn=-1) #Turn off and on the error message for a moment because there is an error caused by the font problem.

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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

options(warn=0)

#below 'tbt_fted' is filtered using word_filter (seperate bigram, filter w1, w2 with word_filter we made once, and unite to bigram again)
tbt_fted <- tr_bigram_token %>%
  separate(bigram, c("w1","w2"), sep=" ") %>%
  filter(!w1 %in% word_filter$word) %>%
  filter(!w2 %in% word_filter$word) %>%
  unite(bigram, w1, w2, sep=" ")

options(warn=-1)

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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

options(warn=0)

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

options(warn=-1)

tr_bigram_token %>%
  filter(bigram %in% class_bigram_tf$bigram) %>%
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
  theme_gray(base_family = "Helvetica") +
  facet_wrap(~ Class, ncol=3, scales="free") +
  coord_flip()

options(warn=0)

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

#training set을 tr과 valid set으로 나눠서 작성해본다. 
set.seed(180302)
sam_num <- sample(nrow(trv), 2200)
ID_list <- sort(unique(trv$ID))
tr_num <- ID_list[sam_num]
te_num <- ID_list[-sam_num]

tr_word_data <- tr_word_token %>%
  filter(ID %in% tr_num) %>%
  count(ID, word) %>%
  merge(trv, by="ID") %>%
  select(ID, word, n, Class)

te_word_data <- tr_word_token %>%
  filter(ID %in% te_num) %>%
  count(ID, word) %>%
  merge(trv, by="ID") %>%
  select(ID, word, n)

te_word_label <- tr_word_token %>%
  filter(ID %in% te_num) %>%
  select(ID, Class) %>%
  mutate(ID=as.numeric(ID)) %>%
  arrange(ID) %>%
  distinct()

tr_bigram_data <- tr_bigram_token %>%
  filter(ID %in% tr_num) %>%
  count(ID, bigram) %>%
  merge(trv, by="ID") %>%
  select(ID, bigram, n, Class)

te_bigram_data <- tr_bigram_token %>%
  filter(ID %in% te_num) %>%
  count(ID, bigram) %>%
  merge(trv, by="ID") %>%
  select(ID, bigram, n)
  

te_bigram_label <- tr_bigram_token %>%
  filter(ID %in% te_num) %>%
  merge(trv, by="ID") %>%
  select(ID, bigram, Class) %>%
  select(ID, Class) %>%
  mutate(ID=as.numeric(ID)) %>%
  arrange(ID) %>%
  distinct()

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
  
  trv_temp <- trv %>%
                filter(ID %in% tr_len$ID)

  tr_gene <- onehot_gene(trv_temp, 30)
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

result_compare <- data.frame(cbind(c(1:9),matrix(nrow=9,ncol=0)))

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

names(result_compare) <- c("Class", "xgb_wd_n_precision", "xgb_wd_n_recall", "xgb_wd_tf_precision", "xgb_wd_tf_recall", "xgb_bg_n_precision", "xgb_bg_n_recall", "xgb_bg_tf_precision", "xgb_bg_tf_recall")
result_compare <- as.data.frame(t(result_compare[,-1]))
names(result_compare) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")

result_compare
