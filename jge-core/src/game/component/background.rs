use super::{component, component_impl, layer::LayerShader, node::Node};
use crate::game::entity::Entity;
use crate::resource::{Resource, ResourceHandle, ResourcePath};

/// 背景组件：可用于渲染纯色或图片背景，并可选叠加自定义片元着色器。
///
/// 渲染规则：每次渲染某个 Layer 时，会在该 Layer 其它内容渲染之前，
/// 先在“以 Layer 所在节点为根的节点树”中按先序遍历找到第一个 Background 并渲染。
///
/// # 示例
///
/// ```no_run
/// use jge_core::game::{
///     component::{background::Background, layer::Layer},
///     entity::Entity,
/// };
///
/// # fn main() -> anyhow::Result<()> {
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async {
///     let layer_root = Entity::new().await?;
///     layer_root.register_component(Layer::new()).await?;
///     layer_root.register_component(Background::new()).await?;
///     Ok::<(), anyhow::Error>(())
/// })?;
/// Ok(())
/// # }
/// ```
#[component(Node)]
#[derive(Debug, Clone)]
pub struct Background {
    entity_id: Option<Entity>,
    color: [f32; 4],
    image: Option<ResourceHandle>,
    fragment_shader: Option<LayerShader>,
}

#[component_impl]
impl Background {
    /// 创建一个默认背景（黑色不透明）。
    #[default()]
    pub fn new() -> Self {
        Self {
            entity_id: None,
            color: [0.0, 0.0, 0.0, 1.0],
            image: None,
            fragment_shader: None,
        }
    }

    /// 设置背景颜色（RGBA，0..1）。
    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    /// 设置背景图片资源（会与 color 相乘作为 tint）。
    pub fn set_image(&mut self, image: Option<ResourceHandle>) {
        self.image = image;
    }

    pub fn image(&self) -> Option<ResourceHandle> {
        self.image.clone()
    }

    /// 设置背景片元着色器（叠加/替换背景绘制逻辑）。
    pub fn set_fragment_shader(&mut self, shader: Option<LayerShader>) {
        self.fragment_shader = shader;
    }

    pub fn fragment_shader(&self) -> Option<&LayerShader> {
        self.fragment_shader.as_ref()
    }

    /// 通过资源路径快速设置片元着色器。
    pub fn set_fragment_shader_from_path(
        &mut self,
        language: super::layer::ShaderLanguage,
        resource_path: ResourcePath,
    ) -> anyhow::Result<()> {
        let resource = Resource::from(resource_path.clone())
            .ok_or_else(|| anyhow::anyhow!("资源路径 {} 未注册", resource_path.join("/")))?;
        self.fragment_shader = Some(LayerShader::new(language, resource));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entity::Entity;

    #[tokio::test(flavor = "multi_thread")]
    async fn background_requires_node_dependency() {
        let entity = Entity::new().await.expect("应能创建实体");
        let _ = entity.unregister_component::<Node>().await;

        let inserted = entity
            .register_component(Background::new())
            .await
            .expect("缺少 Node 时应自动注册依赖");
        assert!(inserted.is_none());
        assert!(entity.get_component::<Node>().await.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn background_updates_color_and_image() {
        let entity = Entity::new().await.expect("应能创建实体");
        entity
            .register_component(Background::new())
            .await
            .expect("应能插入 Background");

        {
            let mut bg = entity.get_component_mut::<Background>().await.unwrap();
            bg.set_color([1.0, 0.5, 0.25, 0.75]);
            bg.set_image(Some(Resource::from_memory(vec![1, 2, 3])));
        }

        let bg = entity.get_component::<Background>().await.unwrap();
        assert_eq!(bg.color(), [1.0, 0.5, 0.25, 0.75]);
        assert!(bg.image().is_some());
    }
}
