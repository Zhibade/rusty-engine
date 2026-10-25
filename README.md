### 日本語は下記です。

# Overview
PBR renderer implemented from scratch in Vulkan that uses a ECS (Entity-Component-System) architecture.

It has the following features:
- GLTF mesh loading
- PNG and EXR image loading
- Automatic texture mipmap generation
- Equirectangular environment map -> Cubemap conversion (compute shader)
- Physically based rendering
  - Irradiance map, prefiltered map, and BRDF's LUT are automatically generated via compute shaders
  - Image-based lighting
  - PBR settings customization via `Material` components
- Tonemapping as a separate subpass
- Post-processing (only subpass-based, for now)
- Modular component-based architecture for easily adding meshes and/or post-processes with custom transforms, shaders, materials, and textures

**NOTE:** This project was created for learning purposes. As such, it might or might not have extra features added in the future.

![Rusty Engine 01](docs/screenshot_01.png)
![Rusty Engine 02](docs/screenshot_02.png)
![Rusty Engine 03](docs/screenshot_03.png)

# Prerequisites
- [Rust](https://rust-lang.org/tools/install/)
  - If using MSVC toolchain in Windows, also need to install MSVC compiler through [Visual Studio](https://visualstudio.microsoft.com/ja/downloads/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

# How to use
- Just run project through IDE or by executing `cargo run` in the command line

## How to add and customize objects
In [app.rs](src/core/app.rs)'s `init_world` method do the following: 
- Load meshes with `mesh_loader`'s [load_mesh](src/loaders/mesh_loader.rs#L15)
- As a minimum, create the following components for each object:
    - [Transform](src/components/transform.rs)
    - [Material](src/components/material.rs)
- Add entity to [World](src/core/world.rs) using `add_entity` method
- Add components to entity using [World](src/core/world.rs)'s `add_component` method

```rust
// Load mesh
let my_mesh = mesh_loader::load_mesh("assets/models/my_mesh.gltf");

// Initialize textures
let albedo_tex = Texture {
    filepath: "assets/textures/my_albedo_tex.png",
    is_cubemap: false,
    usage: TextureUsage::Color,
};

let normal_tex = Texture {
    filepath: "assets/textures/my_normal_tex.png",
    is_cubemap: false,
    usage: TextureUsage::Normal,
};

let orm_tex = Texture {
    filepath: "assets/textures/my_orm_tex.png",
    is_cubemap: false,
    usage: TextureUsage::OcclusionRoughnessMetallic,
};

// Create material and specify parameters
let my_material = Material {
    name: "MyMaterial".to_string(),
    textures: vec![albedo_tex, normal_tex, orm_tex],
    pbr_parameters: Some(PbrParameters {
        color_tint: [0.82, 0.76, 0.09, 1.0],
        roughness_strength: 0.75,
        ..Default::default()
    }),
    ..Material::default_pbr()
};

// Create transform to set position
let my_transform = Transform::from_position(Vec3::from_array([0.0, 0.5, -1.0]));

// Add entity and components
let my_entity_id = world.add_entity();
world.add_component(my_entity_id, my_mesh);
world.add_component(my_entity_id, my_material);
world.add_component(my_entity_id, my_transform);
```

## Processing/changing components at runtime
Since the project uses an ECS architecture, to process or change components at runtime it is necessary to
create a struct that implements the [System](src/systems/system.rs) trait and its `update` method (components
are only data containers).

Inside the `update` method, you can use the [World](src/core/world.rs)'s `query`, `query_pair`, and `query_trio`
methods to get the necessary component combinations and mutate them.

```rust
for (mesh, transform) in world.query_pair::<Mesh, Transform>()
{
    let mut transform_mut = transform.borrow_mut();

    // Do processing
}
```

Once implemented, just need to register the system using [World](src/core/world.rs)'s `register_system`
method.

## Shader Compilation
Since shaders use `#include` directive, they have to be compiled using *glslc.exe* like below

`"<VULKAN_SDK>\Bin\glslc.exe" "my_shader.vert" -o "my_shader_vert.spv" -c`

# No AI usage disclaimer
As this project was created for the purpose of learning, obviously **no AI was used during its implementation
and design.**

# Third Party Assets
**Mesh:** Material ball exported from [3DCoat](https://3dcoat.com)

**Environment Texture:**
https://polyhaven.com/a/charolettenbrunn_park

## Assets used in the screenshots
**Metal textures:** https://freepbr.com/product/scuffed-metal1/  
**Wood textures:** https://freepbr.com/product/bamboo-wood-pbr-material/  
**Other environment texture:** https://polyhaven.com/a/church_meeting_room

---

# 概要
ECS設計を利用したVulkanのPBRレンダラー

機能:
- GLTFメッシュの読み込み
- PNGとEXRの画像読み込み
- テクスチャーミップマップの自動生成
- 正方形図法の環境マップからキューブマップへの変換（コンピュートシェーダー）
- PBR（物理ベースレンダリング）
  - Irradiance Map、Prefiltered Map、 BRDFのLUTの自動生成（コンピュートシェーダー）
  - IBL（イメージベースドライティング）
  - `Material`コンポネントでのPBR設定調整
- トーンマッピング（Subpass）
- ポストプロセス（現在はSubpassのみ）
- モジュラーコンポネントでメッシュ又はポストプロセスを簡単に追加し、カスタムのトランスフォーム、シェーダー、マテリアル、テクスチャを設定できます

**注意:** 当プロジェクトは学習・勉強のために作成されたものですので、将来的には変更・更新がない可能性があります。

![Rusty Engine 01](docs/screenshot_01.png)
![Rusty Engine 02](docs/screenshot_02.png)
![Rusty Engine 03](docs/screenshot_03.png)

# 必要なソフト
- [Rust](https://rust-lang.org/tools/install/)
    - WindowsのMSVCツールチェーンのご利用の場合は[Visual Studio](https://visualstudio.microsoft.com/ja/downloads/)でインストール必要があります
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

# 使い方
- IDEで実行するか、コマンドラインで`cargo run`を実行します

## オブジェクトの追加・変更
[app.rs](src/core/app.rs)の`init_world`関数で下記を行います:
- メッシュを`mesh_loader`の[load_mesh](src/loaders/mesh_loader.rs#L15)でロードします
- 最低限としては下記のコンポネントが必要です:
    - [Transform](src/components/transform.rs)
    - [Material](src/components/material.rs)
- Entityを[World](src/core/world.rs)の`add_entity`で追加します
- コンポネントを[World](src/core/world.rs)の`add_component`で追加します

```rust
// メッシュをロード
let my_mesh = mesh_loader::load_mesh("assets/models/my_mesh.gltf");

// テクスチャーを定義
let albedo_tex = Texture {
    filepath: "assets/textures/my_albedo_tex.png",
    is_cubemap: false,
    usage: TextureUsage::Color,
};

let normal_tex = Texture {
    filepath: "assets/textures/my_normal_tex.png",
    is_cubemap: false,
    usage: TextureUsage::Normal,
};

let orm_tex = Texture {
    filepath: "assets/textures/my_orm_tex.png",
    is_cubemap: false,
    usage: TextureUsage::OcclusionRoughnessMetallic,
};

// マテリアル作成とPBRパラメーター調整
let my_material = Material {
    name: "MyMaterial".to_string(),
    textures: vec![albedo_tex, normal_tex, orm_tex],
    pbr_parameters: Some(PbrParameters {
        color_tint: [0.82, 0.76, 0.09, 1.0],
        roughness_strength: 0.75,
        ..Default::default()
    }),
    ..Material::default_pbr()
};

// トランスフォームで配置します
let my_transform = Transform::from_position(Vec3::from_array([0.0, 0.5, -1.0]));

// Entityとコンポネント追加
let my_entity_id = world.add_entity();
world.add_component(my_entity_id, my_mesh);
world.add_component(my_entity_id, my_material);
world.add_component(my_entity_id, my_transform);
```

## ランタイムでのオブジェクト処理
当プロジェクトはECS設計が利用されていますので、ランタイムでコンポネントを処理するため、
[System](src/systems/system.rs)のトレイトの`update`関数を実装する構造体が必要です
（コンポネントはデータコンテナのみになります）。

`update`の中で[World](src/core/world.rs)の`query`、`query_pair`、`query_trio`
という関数を利用し必要なコンポネント組み合わせを取得し変更できます。

```rust
for (mesh, transform) in world.query_pair::<Mesh, Transform>()
{
    let mut transform_mut = transform.borrow_mut();

    // 処理
}
```

Systemの実装ができたら、[World](src/core/world.rs)の`register_system`関数でSystemを登録します。

## シェーダーコンパイル
シェーダーは`#include`のディレクティブが利用されていますので, 下記のように*glslc.exe*でコンパイルする必要があります。

`"<VULKAN_SDK>\Bin\glslc.exe" "my_shader.vert" -o "my_shader_vert.spv" -c`

# AIは不使用
当然ですが、当プロジェクトは学習・勉強のためのものですので、**設計と実装ではAIは完全に不使用です。**

# サードパーティーのアセット
**メッシュ:** [3DCoat](https://3dcoat.com)からのマテリアルボール

**環境マップ：**
https://polyhaven.com/a/charolettenbrunn_park

## スクリーンショットで利用されたアセット
**金属テクスチャー:** https://freepbr.com/product/scuffed-metal1/  
**木材テクスチャー:** https://freepbr.com/product/bamboo-wood-pbr-material/  
**その他の環境マップ:** https://polyhaven.com/a/church_meeting_room