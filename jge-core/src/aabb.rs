//! Axis-Aligned Bounding Box (AABB) utilities.
//!
//! This module provides simple `Aabb2`/`Aabb3` types for common engine use cases.

use nalgebra::{Vector2, Vector3};

/// 2D axis-aligned bounding box in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb2 {
    pub min: Vector2<f32>,
    pub max: Vector2<f32>,
}

impl Default for Aabb2 {
    fn default() -> Self {
        let origin = Vector2::new(0.0, 0.0);
        Self {
            min: origin,
            max: origin,
        }
    }
}

impl Aabb2 {
    /// Creates an AABB from two points.
    ///
    /// The resulting box is normalized so that `min` is component-wise <= `max`.
    pub fn new(a: Vector2<f32>, b: Vector2<f32>) -> Self {
        let min = Vector2::new(a.x.min(b.x), a.y.min(b.y));
        let max = Vector2::new(a.x.max(b.x), a.y.max(b.y));
        Self { min, max }
    }

    pub fn center(&self) -> Vector2<f32> {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vector2<f32> {
        self.max - self.min
    }

    pub fn contains_point(&self, p: Vector2<f32>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    pub fn contains_aabb(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vector2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Vector2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }

    /// Returns a box expanded by `amount` on all sides.
    pub fn expanded(&self, amount: f32) -> Self {
        let delta = Vector2::new(amount, amount);
        Self {
            min: self.min - delta,
            max: self.max + delta,
        }
    }
}

/// 3D axis-aligned bounding box in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl Default for Aabb3 {
    fn default() -> Self {
        let origin = Vector3::new(0.0, 0.0, 0.0);
        Self {
            min: origin,
            max: origin,
        }
    }
}

impl Aabb3 {
    /// Creates an AABB from two points.
    ///
    /// The resulting box is normalized so that `min` is component-wise <= `max`.
    pub fn new(a: Vector3<f32>, b: Vector3<f32>) -> Self {
        let min = Vector3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Vector3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));
        Self { min, max }
    }

    pub fn center(&self) -> Vector3<f32> {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    pub fn contains_point(&self, p: Vector3<f32>) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    pub fn contains_aabb(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
            && self.max.z >= other.max.z
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Returns a box expanded by `amount` on all sides.
    pub fn expanded(&self, amount: f32) -> Self {
        let delta = Vector3::new(amount, amount, amount);
        Self {
            min: self.min - delta,
            max: self.max + delta,
        }
    }
}
