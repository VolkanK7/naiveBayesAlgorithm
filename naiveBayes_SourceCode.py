#Bu programda, veri setindeki hastaların diyabetli olup olmadığını tahmin etmek için
#..Naive Bayes sınıflandırma algoritması uygulandı.
#Modeli değerlendirmek için sklearn kütüphanesi kullanılarak oluşturulan doğruluk ve sınıflandırma raporu elde edildi.

#İlk olarak, pandas kütüphanesi içe aktarıldı ve ardından
#veri setimizi içeren diabetes dosyası pd.read_csv () komutu ile okundu.
import pandas as pd
data = pd.read_csv("diabetes.csv")

#Verinin ilk 10 satırını yazdırmak için .head (10) komutu kullanıldı.
print("\n\t\t\t-------------------- Veri Seti ----------------------")
print(data.head(10))

#Diabet veri setindeki istatistiklerin bir özetini almak için data.describe() komutu kullanıldı.
#Bu komut ile verilerinin dağılımı hakkında bilgi sahibi olmuş olduk.
print("\n\t\t------------ Veri Seti Özeti --------------")
print(data.describe())

#Veri sütunlarını bağımlı ve bağımsız değişkenlere ayırıldı(x bağımlı, y bağımsız değişken).
#Daha sonra bu değişkenler eğitim ve test veri setine ayrıldı.
#Bunun için sklearn kütüphanesinden train_test_split import edildi.
from sklearn.model_selection import train_test_split
x = data.drop("Outcome", axis=1)
y = data[["Outcome"]]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30, random_state=1)

#Verileri eğitim ve teste böldükten sonra, eğitim setinde bir Naive Bayes modeli oluşturuldu.
#Daha sonra test veri setlerinde tahmin gerçekleştirildi.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)

#Modeli değerlendirmek için, gerçek ve tahmin edilen değerleri kullanarak doğruluğu kontrol edildi.
#Öncelikle sklearn kütüphanesinden doğruluk hesaplaması için metrikler import edildi ve sınıflandırıcının diyabetli kişiyi
#..ne sıklıkla doğru şekilde tahmin ettiği kontrol edildi.
from sklearn import metrics
print("\nACCURACY:", metrics.accuracy_score (y_test,y_pred))

#Naive Bayes modelinden gelen tahmin kalitesini ölçmek için bir sınıflandırma raporu oluşturuldu.
print("\n\t\t------------ Model Kalite Raporu --------------")
test_pred = model.predict(x_test)
print(metrics.classification_report(y_test, test_pred))
print(metrics.confusion_matrix(y_test, test_pred))

#Bulunan tamin kalitesi sonuçlarında;
#Precision, olumlu tahminlerin doğruluk oranını göstermektedir.
#Recall, olumlu vakaların yüzde kaçını yakalayabildiğimizi göstermektedir.
#F1 Score, olumlu vaka tahmininin yüzde kaçının doğru olduğunu göstermektedir.