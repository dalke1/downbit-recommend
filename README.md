# 视频推荐系统

这是一个基于 FastAPI 和 SentenceTransformer 的视频推荐系统。系统通过用户标签、喜欢的视频、收藏的视频以及关键词来生成推荐视频列表。

## 功能介绍

- **用户标签推荐**：根据用户对不同标签的兴趣生成推荐。
- **关键词推荐**：根据用户输入的关键词生成推荐。

## 使用指南

### 1. 克隆项目,并启动服务

```bash
git clone https://github.com/dalke1/downbit-recommend.git

cd downbit-recommend

pip install -r requirements.txt

python recommendSystem.py
```

### 2. 使用接口

向 /recommend/ 端点发送 POST 请求，数据格式如下：

```json
{
    "userModel": {
        "游戏": 3.0,
<<<<<<< HEAD
        "音乐": 1.0,    //与keyWords二选一
=======
        "音乐": 1.0,    //与keyWords二选一
>>>>>>> 617a7cf (实现了关键词推荐和用户模型推荐)
        "电影": 2.0
    },
    "videos": {
        "视频1": ["游戏", "竞技"],
<<<<<<< HEAD
        "视频2": ["音乐", "演唱"],  //必要数据
        "视频3": ["电影", "剧情"]
    },
    "likeVideos": ["视频4"], //可选数据
    "favoriteVideos": ["视频5"], //可选数据
    "recommend_count": 5,   //可选数据，默认为5
    "keyWords": ["游戏"]    //与userModel二选一
=======
        "视频2": ["音乐", "演唱"],  //必要数据
        "视频3": ["电影", "剧情"]
    },
    "likeVideos": ["视频4"], //可选数据
    "favoriteVideos": ["视频5"], //可选数据
    "recommend_count": 5,   //可选数据，默认为5
    "keyWords": ["游戏"]    //与userModel二选一
>>>>>>> 617a7cf (实现了关键词推荐和用户模型推荐)
}
```