import xml.etree.ElementTree as ET
import json
import os

# Читаем XML файл
print("Читаем feed.xml...")
tree = ET.parse('feed.xml')
root = tree.getroot()

# Ищем все товары (тег offer)
offers = root.findall('.//offer')

# Создаем папку для сохранения, если ее нет
os.makedirs('examples/products', exist_ok=True)

# Берем первые два товара
for i, offer in enumerate(offers[:2], start=1):
    name = offer.findtext('name', default='Без названия').strip()
    vendor = offer.findtext('vendor', default='').strip()
    description = offer.findtext('description', default='').strip()
    
    # Собираем характеристики из тегов param
    specs = {}
    for param in offer.findall('param'):
        param_name = param.get('name', '').strip()
        param_value = param.text.strip() if param.text else ''
        if param_name and param_value:
            specs[param_name] = param_value

    # Формируем JSON
    product_data = {
        "name": name,
        "manufacturer": vendor,
        "specs": specs,
        "description": description
    }

    # Сохраняем в файл
    filename = f"examples/products/product_{i}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(product_data, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранен {filename}: {name}")