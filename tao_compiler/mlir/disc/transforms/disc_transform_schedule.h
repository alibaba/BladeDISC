/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DISC_TRANSFORMS_TRANSFORM_SCHEDULE_H_
#define DISC_TRANSFORMS_TRANSFORM_SCHEDULE_H_

#include <string>
#include <unordered_map>

#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"

namespace mlir {
namespace disc_ral {

// PatternKind reprensets the category of a given schedule.
// For the same `PatternKind`, we may still have different schedule strategies
// for different shape range. We further use different tags to distinguish such
// schedules within the same category.
enum class PatternKind : int32_t { kNone, kGEMM };

// Converts a pattern kind to its string representation.
std::string patternKindToString(PatternKind kind);

// Creates a pattern kind from its string representation.
PatternKind patternKindFromString(const std::string& str);

// PatternDescription collects everything needed to assign schedule for a give
// fusion pattern.
class PatternDescription {
 public:
  explicit PatternDescription(lmhlo::FusionOp op, FusionPattern& fusionPattern,
                              ShapeAnalysis& shapeAnalysis);

  // Returns the kind of this `PatternDescription`.
  PatternKind getPatternKind() const;

  // Returns the tags attached to this fusion pattern.
  std::string getPatternTagStr() const;

  // Returns the full pattern kind str + tag str.
  std::string getTaggedPatternStr() const;

 private:
  lmhlo::FusionOp op_;
  FusionPattern& fusionPattern_;
  ShapeAnalysis& shapeAnalysis_;
  PatternKind patternKind_;
};

// Factory used to assign specific schedule for the given PatternDescription
using ScheduleFactory =
    std::function<LogicalResult(PatternDescription&, ModuleOp)>;

// A registry for different schedule factories.
class ScheduleFactoryRegistry {
 public:
  // Returns the singleton
  static ScheduleFactoryRegistry& get();

  // Inserts the new `ScheduleFactory`. Returns true if inserted, otherwise
  // false.
  bool registerScheduleFactory(PatternKind kind, const std::string& tag,
                               ScheduleFactory);

  // Returns the schedule factory factor according to `kind` and `tag`.
  // Returns nullptr if not found.
  ScheduleFactory getScheduleFactory(PatternKind kind, const std::string& tag);

 private:
  ScheduleFactoryRegistry() = default;
  std::unordered_map<PatternKind,
                     std::unordered_map<std::string, ScheduleFactory>>
      factoryMap_;
};

// Macros used to define disc transform schedule factory.
#define DISC_TRANSFORM_SCHEDULE(kind, tag, ...) \
  DISC_TRANSFORM_SCHEDULE_UNIQ_HELPER(kind, tag, __COUNTER__, __VA_ARGS__)

#define DISC_TRANSFORM_SCHEDULE_UNIQ_HELPER(kind, tag, ctr, ...) \
  DISC_TRANSFORM_SCHEDULE_UNIQ(kind, tag, ctr, __VA_ARGS__)

#define DISC_TRANSFORM_SCHEDULE_UNIQ(kind, tag, ctr, ...) \
  static bool unused_ret_val_##ctr =                      \
      ::mlir::disc_ral::ScheduleFactoryRegistry::get()    \
          .registerScheduleFactory(kind, tag, __VA_ARGS__);

// Assign schedule for the given PatternDescription according to its kind and
// tag.
class ScheduleDispatcher {
 public:
  // Users may override the schedule by providing its own implementation and
  // pass the schedule files to the dispatcher.
  // Format of `transformFileName`:
  //  "<kind-0>:<tag-str-0>:<filename-0>;<kind-1>:<tag-str-1>:<filename-1>;"
  explicit ScheduleDispatcher(const std::string& transformFileName);

  // Attaches a schedule for the given pattern description.
  LogicalResult dispatch(PatternDescription& pd, ModuleOp m);

 private:
  // Parses schedule modules from the given files.
  LogicalResult parseModuleFromFile(MLIRContext* ctx);
  // Returns true when applied, otherwise false.
  bool tryToApplyScheduleFromParsedFile(PatternDescription& pd, ModuleOp m);

 private:
  std::string transformFileName_;
  // <pattern-kind, <tag-str, module-op>>
  std::unordered_map<PatternKind,
                     std::unordered_map<std::string, OwningOpRef<ModuleOp>>>
      parsedModuleMap_;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TRANSFORMS_TRANSFORM_SCHEDULE_H_
