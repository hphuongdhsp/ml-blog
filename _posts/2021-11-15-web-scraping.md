---
layout: post
title: Hướng dẫn sủ dụng gói thư viện BeautifulSoup trong Python để  Web Scraping
---

Nội dung của bài viết nhằm hướng dẫn các bạn mới làm quen với đào ảnh từ internet bằng gói thư viện beautiful soup. 

Bài viết gồm các ý chính chính sau:

## Outline

- <a href='#1'>1. Giới thiệu về Web Scaping</a>

- <a href='#2'>2. Tổng quan cấu trúc web </a>  
    - <a id='#2-1'>2.1. HTML, CSS </a> 
    - <a id='#2-1'>2.2. Cấu trúc web  </a> 
- <a href='#3'>3. Bài toán cụ thể: craping ảnh từ trang web: http://www.globalskinatlas.com/diagindex.cfm </a>
    - <a id='#3-1'>3.1. Cấu trúc của trang web </a>
    - <a id='#3-2'>3.2. Cách download ảnh thủ công </a> 
    - <a id='#3-2'>3.3. Xây dựng thuật toán từ thác tác down ảnh thủ công  </a> 
- <a href='#4'>4. Giới thiệu gói thư viện beautiful soup </a>
- <a href='#4'>5. Hoàn thành code </a>


# <a id='1'>1.Giới thiệu về Web Scaping</a>

Internet có nguồn dữ liệu khổng lồ, dữ liệu mà chúng ta hoàn toàn có thể truy cập bằng cách sử dụng web cùng một công cụ lập trình (Python, C++). Web Scaping là tác vụ download tất cả thông tin liên quan từ một trang web cố định. Ví dụ chúng ta muốn download tất cả các ảnh từ trang web http://www.globalskinatlas.com/diagindex.cfm để làm phong phú kho dữ liệu. 

Một số trang web cung cấp cho chúng ta thông qua một API (Application Programming Interface), một số trang web khác có thể co ngừoi dùng lấy dữ liệu thông qua database có sẵn. Ví dụ khi bạn muốn download ảnh từ một trang web, bạn click vào ảnh trên website, từ website sẽ đưa bạn tới một trang web khác, nơi đó có lưu trữ ảnh trực tiếp trên server. 

# <a id='2'>2. Tổng quan cấu trúc web</a>

Trước khi đi sâu vào làm sao có thể download tất cả dữ liệu từ một trang web, chúng ta sẽ tìm hiểu cấu trúc của một trang web. Việc này giống như đi câu cá, bạn tìm hiểu cấu trúc của hồ nước, để có thêm thông tin giúp việc câu cá dễ dàng hơn. 


## <a id='2.1'>2.1 Tổng quan HTML, CSS</a>

Khi chúng ta truy cập một trang web, trình duyệt web (Firefox, Chrome) đưa ra yêu cầu đến máy chủ của trang web. Yêu cầu này được gọi là yêu cầu GET, sau đó chúng ta nhận được thông tin từ máy chủ. Nguồn thông tin từ máy chủ sẽ vẫn được trả lại thông tin gồm những tập file. Nhờ trình duyệt web, các tập này sẽ hiển thị dứoi dạng web. Cấu thành của tập để trình duyệt web có thể đọc một trang web bao gồm:

HTML - nội dung chính của trang.
CSS - File này hỗ trợ HTML để hiển thi web đẹp hơn.
JS - Các tệp Javascript thêm tính tương tác cho các trang web.
Hình ảnh - các định dạng hình ảnh, chẳng hạn như JPG và PNG, cho phép các trang web hiển thị hình ảnh.
Sau khi trình duyệt của chúng tôi nhận được tất cả các tệp, nó sẽ hiển thị trang và hiển thị cho chúng tôi.

Ví dụ: Khi chúng ta vào trình duyệt Chrome, chúng ta muốn try tập vào trang http://www.globalskinatlas.com/diagindex.cfm , khi đó máy chủ sẽ trả lại một tập, tập dữ liệu này gồm các file (html, css, javascript,), các file này sẽ được gửi trực tiếp về Chrome, thông qua trình duyệt, tất cả các tệp này sẽ tạp nên một trang web.  

Để hiểu rõ cấu trúc một trang web, chúng ta sẽ tìm hiểu sâu file HTML. Ở các trình duyệt. Để hiển thị cấu trúc file HTML, chúng ta bấm phím *F12*. 

## <a id='2.2'>2.2 Tổng quan HTML</a>

Cấu trúc cơ bản của trang HTML có dạng như sau: 

    - <!Doctype>: Phần khai báo chuẩn của html hay xhtml.
    - <head></head>: Phần khai báo ban đầu, khai báo về meta, title, css, javascript…
    - <body></body>: Phần chứa nội dung của trang web, nơi hiển thị nội dung.
  
```
<!DOCTYPE html>
<html>
<head>
<title>Phần tiêu đề của html </title>
</head>

<body>
...Phần thân của html ...
</body>
</html>
```
Ở phần tiếp theo chúng ta sẽ giới thiệu về thẻ liên kết \<a\>, một những phần quan trọng nhất để thực hành đào ảnh. 
## <a id='2.2.1'>2.2.1  Thẻ liên kết *a* 

    - Thẻ liên kết \<a\> \</a> dùng để tạo một liên kết từ trang web này sang trang web khác, từ vị trí này sang vị trí khác hay dùng để mở ra một object nào đó (có thể là file words, ảnh, excel, pdf, mp3, movie,...), thẻ này có một thuộc tính bắt buộc:

    - href: Chứa đường dẫn cụ thể tới mục tiêu liên kết.

Ví dụ: Trong trang web [http://www.globalskinatlas.com/diagdetail.cfm?id=91](http://www.globalskinatlas.com/diagdetail.cfm?id=91),  khi chúng ta sử dụng phím f12, một trong những tag  \<a\> \</a\> có dạng như sau


```
<a href="imagedetail.cfm?TopLevelid=170&amp;ImageID=462&amp;did=8 ">View</a>
```

    - Bằng truy cập trang web, ta thấy được liên kết ở tag này là: http://www.globalskinatlas.com/imagedetail.cfm?TopLevelid=170&ImageID=462&did=8 
    - Text để mô tả tag này là *View*
  
## <a id='2.2.2'>2.2.2  Thẻ liên kết *img* 

    - Thẻ hiển thị một image \<*img*/> dùng để nhúng một ảnh thông qua thuộc tính src, thẻ này có 2 thuộc tính bắt buộc:
    - src: Chứa đường dẫn tham chiếu tới image.
    - alt: Được sử dụng như một văn bản thay thế khi image không hiển thị (hoặc không có sẵn).
  
Cấu trúc của thẻ <*img*> không có sử dụng thẻ đóng (không dùng <*img*></*img*>), mà sử dụng ký tự kết thúc là một khoảng trắng và ký tự "/".
Tham khảo thêm về thẻ <*img*/>.
# <a id='3'> 3.Bài toán cụ thể: craping ảnh từ trang web: http://www.globalskinatlas.com/diagindex.cfm</a>

Ở phần này chúng ta sẽ đi sâu vào phân tích cụ thể và định hướng hướng làm. 

Đầu bài: Download tất cả ảnh có ở trang web http://www.globalskinatlas.com/diagindex.cfm và lưu trữ ảnh đó ở thư mục phù hợp. 

Nhiệm vụ xuất phát của bài toán xuất phát từ nhu cầu thu thập ảnh bệnh nhân bị bệnh và chúng ta muốn lưu trữ thông tin các bệnh của từng ảnh để tiện sau này sử dụng cho các mô hình  về  machine learning. 


## <a id='3.1'> 3.1 Cấu trúc của trang web</a>
Chúng ta cùng xem xét cấu trúc của trang web. Khi thực hiện vào trang web, màn hình sẽ hiện thị như ảnh bên dưới. Chúng ta có thể thấy file được có rất nhiều tag <*a*></*a*>, mỗi tag tương ứng với một bệnh. 



<img align="center" width="600"  src="https://habrastorage.org/webt/27/j-/gq/27j-gqhx7wj_owger_2hzxjdt9a.png">



Bấm phím f12 để thấy được cấu trúc của trang web. Mỗi tag bệnh sẽ tương ứng với

```
<a href="diagdetail.cfm?id=653"></a>
```

Nếu click vào một tag, sẽ đưa chúng ta tới một trang web mới, ví dụ ở đây chúng ta click vào *Ecthyma*, chúng ta được tới trang web http://www.globalskinatlas.com/diagdetail.cfm?id=653 . 

Nhận thấy rằng diagdetail.cfm?id=653 sẽ là tương ứng với bệnh Ezthyma. Và id *653* sẽ tương ứng với mã bệnh *Ecthyma*. Dispay của trang web sẽ có dạng hình như sau 

<img align="center" width="600"  src="https://habrastorage.org/webt/yb/je/f_/ybjef_oy1pkacarf64jqdthvysu.png">


Khi click vào View ở góc cuối cùng, chúng ta sẽ được tới một trang web mới : http://www.globalskinatlas.com/imagedetail.cfm?TopLevelid=1099&ImageID=2615&did=6

Trang web mới này được gắn ở href của một tag <*a*></*a*> của trang web http://www.globalskinatlas.com/diagdetail.cfm?id=653 , cụ thể  nội dung của tag <*a*></*a*>: 

```
<a href="imagedetail.cfm?TopLevelid=1099&amp;ImageID=2615&amp;did=6">View</a>
```

Có thể thấy "imagedetail.cfm?TopLevelid=1099&amp;ImageID=2615&amp;did=6" là tag gắn liền với trang web http://www.globalskinatlas.com/imagedetail.cfm?TopLevelid=1099&ImageID=2615&did=6. 

Hiện thị của trang web sẽ có dạng như sau 

<img align="center" width="600"  src="https://habrastorage.org/webt/13/vc/9n/13vc9ncojyhrrwaqbzkes1bwigq.png">


Ở trang web này, khi nhấp chuột phải, chúng ta hoàn toàn có thể download ảnh thủ công. Nhưng khi tiến hành download ảnh, chúng ta nhận ra rằng, chỉ có ảnh ở trung tâm có size ảnh lớn, những ảnh nhỏ hơn sẽ có size nhỏ hơn. 
Vì vậy chúng ta sẽ tiến hành download ảnh ở trung tâm, còn với mỗi ảnh nhỏ ở dưới, chúng ta sẽ nhấn click chuột vào ảnh, ví dụ chúng ta sẽ click vào ảnh nhỏ đầu tiên, từ link ảnh nhỏ sẽ đưa tới trang web có display như sau 

<img align="center" width="600"  src="https://habrastorage.org/webt/ck/hp/1-/ckhp1-nveji7id0o10lmkxgsr9s.png">
 
Chúng ta sẽ tiếp tục download ảnh ở trung tâm và lưu ở thư mục bệnh. 


## <a id='3.2'> 3.2 Thuật toán đào ảnh thủ công</a>

Dựa những phân tích như ở trên, chúng ta xây dựng được thuật toán đào ảnh như sau 

- 1. Truy cập trang web http://www.globalskinatlas.com/diagindex.cfm
  - 1.1 Truy cập tag bệnh đầu tiên , tạo thư mục bệnh đầu tiên
    - 1.1.1 Truy vào tag View đầu tiên 
      - 1.1.1.0 Tải ảnh trung tâm lưu vào folder bệnh 
      - 1.1.1.1 Truy vào ảnh nhỏ đâù tiên và tải ảnh trung tâm lưu vào folder bệnh
      - 1.1.1.2 Truy vào ảnh nhỏ thứ hai  và tải ảnh trung tâm lưu vào folder bệnh
    - 1.1.2 Truy vào tag View thứ hia 
      - 1.1.2.0 Tải ảnh trung tâm lưu vào folder bệnh 
      - 1.1.2.1 Truy vào ảnh nhỏ đâù tiên và tải ảnh trung tâm lưu vào folder bệnh
      - 1.1.2.2 Truy vào ảnh nhỏ thứ hai  và tải ảnh trung tâm lưu vào folder bệnh
    - ... 
  - 1.2 Truy cập tag bệnh đầu tiên , tạo thư mục bệnh đầu tiên
    - 1.2.1 Truy vào tag View đầu tiên 
      - 1.2.1.0 Tải ảnh trung tâm lưu vào folder bệnh 
      - 1.2.1.1 Truy vào ảnh nhỏ đâù tiên và tải ảnh trung tâm lưu vào folder bệnh
      - 1.2.1.2 Truy vào ảnh nhỏ thứ hai  và tải ảnh trung tâm lưu vào folder bệnh
    - 1.2.2 Truy vào tag View thứ hia 
      - 1.2.2.0 Tải ảnh trung tâm lưu vào folder bệnh 
      - 1.2.2.1 Truy vào ảnh nhỏ đâù tiên và tải ảnh trung tâm lưu vào folder bệnh
      - 1.2.2.2 Truy vào ảnh nhỏ thứ hai  và tải ảnh trung tâm lưu vào folder bệnh
    - ... 


Thuật toán trên đảm bảo giúp chúng ta có thể tải tất cả các ảnh và lưu trữ ảnh ở thư mục phù hợp. Nhiệm vụ của chúng ta bước tiếp theo sẽ tìm hiểu gói thưu viện beautiful soup và thực hiện hoá thuật toán thủ công 


## <a id='4'> 4 Giới thiệu gói thư viện beautiful soup </a>


BeautifulSoup là một gói thư viện của Python nhằm giúp người dùng dễ  dàng lấy dữ liệu ra khỏi các file HTML và XML. Bạn đọc có thể tham khảo chi tiết ở trang web https://www.howkteam.vn/d/thu-vien-beautiful-soup-460. 

Trong khuôn khô bài viết, chúng ta sẽ tìm hiểu những lệnh cơ bản sau: 

### Lệnh khởi tạo soup
```
r = requests.get(web_url)
soup = BeautifulSoup(r.content, "html.parser")
```
### Lệnh lấy tất cả \<a> tag 
Lệnh trên nhằm tạo món soup dựa trên nguyên liệu trang web *web_url*, giúp bạn dễ dàng hơn trong việc truy cập dữ liệu của file HTML *web_url*

```
links = soup.findall("a", href = Trues)
```
### Lấy url ở trong một tag 
Lệnh giúp bạn lấy url trong một tag <*a*></*a*> từ món soup có sẵn. 

```python
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0L1x0DM4mpSt97%2ftYgbxlC2B7n4pvJNhhvRwo8bxiO4B" class="sister" id="link1">Elsie</a>,
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0OPun6GIXb9bh0UOloN9WCYbJtHZQd%2fvB08D2UeudkPP" class="sister" id="link2">Lacie</a> and
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0LirHL60gbBHH3VIishi9CqgtHAKmbGoKNvFheNkumnh" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')

for a in soup.find_all('a', href=True):
    print ("Found the URL:", a['href'])
```

Out put trả ra:

```
Found the URL: /redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0L1x0DM4mpSt97%2ftYgbxlC2B7n4pvJNhhvRwo8bxiO4B
Found the URL: /redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0OPun6GIXb9bh0UOloN9WCYbJtHZQd%2fvB08D2UeudkPP
Found the URL: /redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0LirHL60gbBHH3VIishi9CqgtHAKmbGoKNvFheNkumnh
```
### Lấy text ở trong một tag 

Để lấy một tag trong một tag từ soup, chúng ta sử dụng lệnh: 
```
tag.text.strip()
```
Ví dụ 

```
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0L1x0DM4mpSt97%2ftYgbxlC2B7n4pvJNhhvRwo8bxiO4B" class="sister" id="link1">Elsie</a>,
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0OPun6GIXb9bh0UOloN9WCYbJtHZQd%2fvB08D2UeudkPP" class="sister" id="link2">Lacie</a> and
<a href="/redirect?Id=f%2fKgPq4IDV0SyEq0zfYr0LirHL60gbBHH3VIishi9CqgtHAKmbGoKNvFheNkumnh" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')

for a in soup.find_all('a', href=True):
    print ("Found the text :", a.text.strip())
```

# <a id='5'>5. Hoàn thành code craping dữ liệu cho bài toán</a>

Từ những lệnh cở bản của *BeautifulSoup* ở mục 4 và thuật toán được xây dựng ở mục 3. Chúng ta xây dựng hàm trên python để craping toàn bộ ảnh bệnh từ trang web http://www.globalskinatlas.com/. 

Source code ở [link](https://github.com/hphuongdhsp/test-beautiful-soup4/blob/main/src/main.py)
