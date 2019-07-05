# -*- coding:UTF-8 -*-

from bs4 import BeautifulSoup
import requests, sys

###HTML:
'''
<div>
<div> 是一个块级元素。这意味着它的内容自动地开始一个新行。实际上，换行是 <div> 固有的唯一格式表现。可以通过 <div> 的 class 或 id 应用额外的样式。
不必为每一个 <div> 都加上类或 id，虽然这样做也有一定的好处。
可以对同一个 <div> 元素应用 class 或 id 属性，但是更常见的情况是只应用其中一种。这两者的主要差异是，class 用于元素组（类似的元素，或者可以理解为某一类元素），而 id 用于标识单独的唯一的元素。

<br>
请使用 <br> 来输入空行，而不是分割段落。

<dl> <dt> <dd>
<dl> 标签定义了定义列表（definition list）。
<dl> 标签用于结合 <dt>(definition terms)定义列表中的项目）和 <dd>(definition describ)（描述列表中的项目）

'''


class downloader(object):
    def __init__(self):
        self.server = 'https://www.biqukan.com/'
        self.target = 'https://www.biqukan.com/1_1094'
        self.proxy = {
            'http' :'http://proxy-shz.intel.com:911',
            'https':'http://proxy-shz.intel.com:911'
        }
        self.names = []  ##章节名
        self.urls = []   ##章节链接
        self.nums = 0    ##章节数

    ## 获取下载链接
    def get_download_url(self):

        req = requests.get(self.target, proxies=self.proxy)
        html = req.text
        div_bf = BeautifulSoup(html,'lxml')
        div = div_bf.find_all('div', class_='listmain')
        a_bf = BeautifulSoup(str(div[0]), 'lxml')
        a = a_bf.find_all('a')  ## html中 <a href=''>存放的是超链接
        self.nums = len(a[15:])  ### 从第一章节开始往后
        for each in a[15:]:
            self.names.append(each.string)
            self.urls.append(self.server + each.get('href'))
    
    ## 获取章节内容
    def get_contents(self, target):
        req = requests.get(url=target, proxies=self.proxy)
        html = req.text
        bf = BeautifulSoup(html, 'lxml')
        texts = bf.find_all('div', class_ = 'showtxt')
        texts = texts[0].text.replace('\xa0'*8, '\n\n')
        return texts
    
    ## 将爬取的内容写入文件
    def writer(self, name, path, text):
        write_flag = True
        with open(path, 'a', encoding='utf-8') as f:
            f.write(name +'\n')
            f.writelines(text)
            f.write('\n\n')

if __name__ == '__main__':
    dl = downloader()
    dl.get_download_url()
    print('开始下载：')
    for i in range(dl.nums):
        dl.writer(dl.names[i], 'C:\Users\lijie4\Documents\GitHub\BasicAlgo\project\一念永恒.txt', dl.get_contents(dl.urls[i]))
        #sys.stdout.write("  已下载: %.3f%%" % float(i/dl.nums) + '\r')
        sys.stdout.flush()
    print('下载完成')



