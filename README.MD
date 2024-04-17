# API

## Query

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded"  -d "query=tự do là gì?" -d "top_k=2" http://127.0.0.1:3322/api/95buddha/chunk/query
```

```json
{
  "matches": [
    {
      "id": "2203e3f8799c41479fdd9667d8b8df56",
      "metadata": {
        "authors": [
          "Nguyên Phong"
        ],
        "document_id": "this-is-test-document-id",
        "pipeline_profile": "intfloat/e5-small",
        "text": "Tự do tư tưởng không phải chỉ là muốn nghĩ thế nào thì nghĩ, mà còn là giải thoát ta ra khỏi các áp lực bắt ta phải suy nghĩ theo một lề lối nào đó.",
        "title": "Hành trình về Phương Đông",
        "types": [
          "Tâm linh"
        ]
      },
      "score": 0.832516849,
      "values": []
    },
    {
      "id": "e3e560deda074acdb22caa90e6b499e4",
      "metadata": {
        "authors": [
          "Paulo Coelho"
        ],
        "document_id": "this-is-test-document-id",
        "pipeline_profile": "intfloat/e5-small",
        "text": "Nếu bạn chinh phục được chính mình, bạn có thể chinh phục được cả thế giới.",
        "title": "Nhà Giả Kim",
        "types": [
          "Tiểu thuyết"
        ]
      },
      "score": 0.832013488,
      "values": []
    }
  ]
}
```

## Upsert (Emb và đẩy dữ liệu lên Pinecone)
authors, types: str (các tên cách nhau bởi dấu , )

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" http://127.0.0.1:3322/api/95buddha/chunk/upsert -d "text=Buổi trưa ăn bưởi chua" -d "authors=sang, san" -d "title=Tác phẩm nhảm nhí" -d "type=Chuyện cười, văn học" 
```

```
# Thành công
{
  "message": "upserted 02cb78f191644ab5aa1560026ea4afcb"
}

# Bị trùng
{
  "message": "duplicated, let's skipping"
}
```


## Delete 1 chunk
```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" http://127.0.0.1:3322/api/95buddha/chunk/delete -d "id=5af07e639fe944bd8de4487bac301d87"
```

```
{
  "message": "Deleted successfully 5af07e639fe944bd8de4487bac301d87"
}
```
