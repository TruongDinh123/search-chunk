# API

## Query

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded"  -d "query=Kho báu đời mình ở đâu" -d "top_k=2" http://127.0.0.1:3322/api/95buddha/query
```

```json
{
  "matches": [
    {
      "id": "03b93cfbc3c846279a069ce73afb52c3",
      "metadata": {
        "authors": [
          "Paullo Coelho"
        ],
        "text": "H\u00e3y nh\u1edb r\u1eb1ng, tr\u00e1i tim c\u1ee7a b\u1ea1n \u1edf \u0111\u00e2u, th\u00ec t\u1ea1i n\u01a1i \u0111\u00f3 b\u1ea1n s\u1ebd t\u00ecm th\u1ea5y kho b\u00e1u c\u1ee7a \u0111\u1eddi m\u00ecnh"
      },
      "score": 0.742982626,
      "values": [
        -0.0879698694,
        0.075806424,
        0.00975498091,
        -0.052055832,
        ...
      ]
    },
    {
      "id": "d6bef45e0edd4c89bfe8a2e7733b91c3",
      "metadata": {
        "authors": [
          "Paullo Coelho"
        ],
        "text": "M\u1ecdi ph\u01b0\u1edbc l\u00e0nh b\u1ecb t\u1eeb ch\u1ed1i \u0111\u1ec1u tr\u1edf th\u00e0nh m\u1ed9t l\u1eddi nguy\u1ec1n"
      },
      "score": 0.738684416,
      "values": [
        -0.0759756342,
        0.0447352901,
        0.0067217024,
        -0.054945454,
        ...
      ]
    }
  ]
}
```

## Upsert (Emb và đẩy dữ liệu lên Pinecone)
```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" http://127.0.0.1:3322/api/95buddha/upsert -d "text=Buổi trưa ăn bưởi chua" -d "authors=sang, san" 
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

