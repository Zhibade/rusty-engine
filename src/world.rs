//! World implementation

use super::components::component::Component;
use super::systems::system::System;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

/// Represents an entity that can contain up to one component of each type
///
/// `dyn Any` represents `RefCell<dyn Component>`\
/// It is stored as `dyn Any` to be able to cast it to its concrete type (`RefCell<T>`)
///
/// TODO: Might be better to handle it as a separate or multiple structs?
type Entity = HashMap<TypeId, Rc<dyn Any>>;

/// Represents a component type combination for caching entities with specific component types
///
/// Using [Vec] instead [HashSet] because it does not implement [Hash] trait, so it needs to be
/// sorted before using as a key or comparing
type ComponentType = Vec<TypeId>;

/// Contains all entities/components present in the "world"
///
/// Index of each entry of `entities` represents the entity ID
pub struct World
{
    entities: Vec<Entity>,

    // Stores entities ID per component combination
    entity_component_cache: HashMap<ComponentType, Vec<usize>>,

    // Only one system instance per system type
    systems: HashMap<TypeId, Box<dyn System>>,
    last_frame_time: Instant,
}

impl World
{
    /// Add a new entity without components to the world
    ///
    /// Entity ID is returned
    pub fn add_entity(&mut self) -> usize
    {
        let new_entity = HashMap::new();
        self.entities.push(new_entity);

        self.entities.len() - 1
    }

    /// Adds the given component to an entity
    ///
    /// # Panics
    /// Will panic if `entity_id` is invalid\
    /// Will panic if component type has not been registered before adding\
    /// Will panic if entity already has component
    ///
    /// TODO: Add version that supports adding multiple components at the same time
    pub fn add_component<T: Component>(&mut self, entity_id: usize, component: T)
    {
        // Check if entity ID is valid
        if entity_id >= self.entities.len()
        {
            panic!("World :: add_component :: Invalid entity ID: {entity_id}");
        }

        let component_type = TypeId::of::<T>();

        // Check if component was already added to entity
        if let Some(_) = self.entities[entity_id].get(&component_type)
        {
            panic!(
                "World :: add_component :: Entity {} already has component of type {:?}.",
                entity_id,
                component.type_id()
            )
        }

        self.entities[entity_id].insert(component_type, Rc::new(RefCell::new(component)));
        self.update_component_cache(entity_id);
    }

    /// Get component of given type from entity
    ///
    /// # Panics
    /// Will panic if `entity_id` is invalid
    pub fn get_component<T: Component>(&self, entity_id: usize) -> Option<Rc<RefCell<T>>>
    {
        // Check if entity ID is valid
        if entity_id >= self.entities.len()
        {
            panic!("World :: get_component :: Invalid entity ID: {entity_id}");
        }

        let component_type = TypeId::of::<T>();

        if self.entities[entity_id].contains_key(&component_type)
        {
            let generic_rc = self.entities[entity_id]
                .get(&component_type)
                .unwrap()
                .clone();

            Some(generic_rc.downcast::<RefCell<T>>().unwrap())
        }
        else
        {
            None
        }
    }

    /// Query for all existing components of the given type
    pub fn query<T: Component>(&self) -> Vec<Rc<RefCell<T>>>
    {
        let mut components = Vec::new();
        let type_id = TypeId::of::<T>();

        let component_cache_key = Vec::from([type_id]);
        if self
            .entity_component_cache
            .contains_key(&component_cache_key)
        {
            let entities_with_component = self
                .entity_component_cache
                .get(&component_cache_key)
                .unwrap();
            entities_with_component.iter().for_each(|entity_id| {
                if let Some(component) = self.get_component::<T>(entity_id.clone())
                {
                    components.push(component);
                }
            });
        }

        components
    }

    /// Query for all existing component pairs of the given types
    pub fn query_pair<T: Component, U: Component>(
        &self,
    ) -> Vec<(Rc<RefCell<T>>, Rc<RefCell<U>>)>
    {
        let mut components = Vec::new();
        let type_id_t = TypeId::of::<T>();
        let type_id_u = TypeId::of::<U>();

        // Component combinations should be cached already, so just use the cache to retrieve them
        let mut component_cache_key = Vec::from([type_id_t, type_id_u]);
        component_cache_key.sort();

        if self
            .entity_component_cache
            .contains_key(&component_cache_key)
        {
            for entity_id in self
                .entity_component_cache
                .get(&component_cache_key)
                .unwrap()
                .iter()
            {
                let component_t = self.get_component::<T>(entity_id.clone());
                let component_u = self.get_component::<U>(entity_id.clone());

                if component_t.is_none() || component_u.is_none()
                {
                    continue;
                }

                components.push((component_t.unwrap(), component_u.unwrap()))
            }
        }

        components
    }

    /// Register a new system type in the world
    ///
    /// System's update method will be run every frame
    ///
    /// Each system type can only be registered once. Call will be ignored if trying to register
    /// an already registered type
    pub fn register_system<T: 'static + System>(&mut self, system: T)
    {
        let type_id = TypeId::of::<T>();

        if !self.systems.contains_key(&type_id)
        {
            self.systems.insert(type_id, Box::new(system));
        }
        else
        {
            #[cfg(debug_assertions)]
            println!(
                "World :: register_system :: System of type {type_id:?} is already registered.\
                Ignoring call."
            );
        }
    }

    /// Updates the component combination caches for this entity
    fn update_component_cache(&mut self, entity_id: usize)
    {
        let mut component_types = Vec::new();

        for component_type in self.entities[entity_id].keys()
        {
            component_types.push(component_type.clone());
            component_types.sort(); // Need to sort to avoid different permutations

            if !self.entity_component_cache.contains_key(&component_types)
            {
                self.entity_component_cache
                    .insert(component_types.clone(), Vec::new());
            }

            self.entity_component_cache
                .get_mut(&component_types)
                .unwrap()
                .push(entity_id);
        }
    }

    /// Update the world for this frame
    ///
    /// All registered systems will run their `update` method
    pub fn update(&mut self)
    {
        let delta_time = (Instant::now() - self.last_frame_time).as_secs_f32();

        self.systems.iter().for_each(|(_, system)| {
            system.update(delta_time, self);
        });

        self.last_frame_time = Instant::now();
    }
}

/// [Default] trait implementation for [World] ([Instant] has no [Default] implementation)
impl Default for World
{
    fn default() -> Self
    {
        World {
            entities: Vec::default(),
            entity_component_cache: HashMap::default(),
            systems: HashMap::default(),
            last_frame_time: Instant::now(),
        }
    }
}
