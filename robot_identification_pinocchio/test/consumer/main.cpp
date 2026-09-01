#include <memory>
#include <type_traits>

#include <robot_identification_pinocchio/pinocchio_regressor.h>

static_assert(std::is_base_of_v<robot_identification::RegressorBackend,
                                robot_identification_pinocchio::PinocchioRegressor>);

int main() { return 0; }
