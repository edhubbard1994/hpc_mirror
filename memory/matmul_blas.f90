  subroutine matmul_blas_f (m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) &
&                                       bind(C,name="matmul_blas")

    use iso_c_binding

    implicit none

    integer(c_int), intent (in), value :: m, n, k, lda, ldb, ldc
    real(c_double), intent (in), value :: alpha, beta
    real(c_double), intent (in)    :: A(*), B(*)
    real(c_double), intent (inout) :: C(*)

    character :: transa = 'N'
    character :: transb = 'N'

    !print *, m, n, k, lda, ldb, ldc, transa, transb
    call dgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc )

  end subroutine

  subroutine dummy_f( x ) bind(C,name="dummy")

    use iso_c_binding

    implicit none

    real(c_double), intent(inout) :: x(*)

    x(1) = x(1)**2

  end subroutine
