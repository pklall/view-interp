#pragma once
#ifndef OPENGM_CUSTOM_FUNCTION_HXX
#define OPENGM_CUSTOM_FUNCTION_HXX

#include <functional>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// Absolute difference between two labels
///
/// \ingroup functions
template<class T, class I = size_t, class L = size_t>
class CustomUnaryFunction
: public FunctionBase<CustomUnaryFunction<T, I, L>, T, I, L>
{
public:
   typedef T ValueType;
   typedef I IndexType;
   typedef L LabelType;

   CustomUnaryFunction(const LabelType, const LabelType,
           std::function<ValueType(LabelType)>);
   size_t shape(const IndexType) const;
   size_t size() const;
   size_t dimension() const;
   template<class ITERATOR> ValueType operator()(ITERATOR) const;

private:
   LabelType numberOfLabels1_;
   std::function<ValueType(LabelType)> func_;
};

/// \cond HIDDEN_SYMBOLS
/// FunctionRegistration
template <class T, class I, class L>
struct FunctionRegistration< CustomUnaryFunction<T, I, L> >{
   /// Id  of the CustomUnaryFunction
   enum ID {
      Id = opengm::FUNCTION_TYPE_ID_OFFSET + 13
   };
};

/// Constructor
///
/// \param numberOfLabels1 number of labels of the first variable
/// \param numberOfLabels2 number of labels of the second variable
///
template <class T, class I, class L>
inline
CustomUnaryFunction<T, I, L>::CustomUnaryFunction
(
   const LabelType numberOfLabels1, 
   std::function<ValueType(LabelType)> func
)
:  numberOfLabels1_(numberOfLabels1), 
   func_(func)
{}

template <class T, class I, class L>
template <class ITERATOR>
inline typename CustomUnaryFunction<T, I, L>::ValueType
CustomUnaryFunction<T, I, L>::operator()
(
   ITERATOR begin
) const {
   return func_(begin[0]);
}

/// extension a value table encoding this function would have
///
/// \param i dimension
template <class T, class I, class L>
inline size_t
CustomUnaryFunction<T, I, L>::shape
(
   const IndexType i
) const {
   OPENGM_ASSERT(i < 1);
   return numberOfLabels1_;
}

// order (number of variables) of the function
template <class T, class I, class L>
inline size_t
CustomUnaryFunction<T, I, L>::dimension() const {
   return 1;
}

/// number of entries a value table encoding this function would have (used for I/O)
template <class T, class I, class L>
inline size_t
CustomUnaryFunction<T, I, L>::size() const {
   return numberOfLabels1_;
}

} // namespace opengm

#endif // OPENGM_ABSOLUTE_DIFFERENCE_FUNCTION_HXX
